import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
import argparse
import time
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import wandb

# Local imports
from datasets.bbox_loader import BBoxLoader
from datasets.random_cube import RandomCubeDataset
from models.simclr3d import SimCLR, nt_xent_loss


class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    See: https://arxiv.org/abs/1708.03888
    
    Adapted from:
    https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
    """
    def __init__(self, params, lr=0.001, weight_decay=0.0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        """
        Args:
            params: Model parameters
            lr: Initial learning rate
            weight_decay: Weight decay for regularization
            momentum: Momentum coefficient
            eta: LARS coefficient
            weight_decay_filter: Function to filter parameters for weight decay
            lars_adaptation_filter: Function to filter parameters for LARS adaptation
        """
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            weight_decay_filter = group['weight_decay_filter']
            lars_adaptation_filter = group['lars_adaptation_filter']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                d_p = p.grad
                
                # Weight decay
                if weight_decay != 0 and (weight_decay_filter is None or weight_decay_filter(p)):
                    d_p = d_p.add(p, alpha=weight_decay)
                
                # LARS adaptation
                if lars_adaptation_filter is None or lars_adaptation_filter(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(d_p)
                    
                    # Avoid division by zero
                    if param_norm != 0 and update_norm != 0:
                        # Compute local learning rate
                        local_lr = eta * param_norm / update_norm
                        # Apply local learning rate
                        d_p = d_p.mul(local_lr)
                
                # Momentum
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                
                # Update parameter
                p.add_(d_p, alpha=-group['lr'])


def setup_distributed():
    """Setup distributed training."""
    # Initialize process group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", world_size=world_size, rank=rank)
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        
        # Set device for current process
        torch.cuda.set_device(device)
        print(f"Initialized process {rank}/{world_size} on device {device}")
        
        return rank, world_size, device
    else:
        # Not using distributed
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, device


def filter_bias_and_bn(model):
    """Get parameter groups for optimizer to exclude BN and bias from weight decay."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Skip BN and bias parameters for weight decay
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay}
    ]


def get_optimizer_and_scheduler(model, lr, weight_decay, epochs, steps_per_epoch):
    """
    Create LARS optimizer and cosine learning rate scheduler.
    
    Args:
        model: SimCLR model
        lr: Learning rate
        weight_decay: Weight decay factor
        epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
        
    Returns:
        Optimizer, scheduler, and gradient scaler
    """
    # Filter parameters for weight decay
    param_groups = filter_bias_and_bn(model)
    
    # Create LARS optimizer
    optimizer = LARS(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
        eta=0.001
    )
    
    # Create cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * steps_per_epoch,
    )
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler()
    
    return optimizer, scheduler, scaler


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, 
                epoch, args, global_step=0):
    """
    Train for one epoch.
    
    Args:
        model: SimCLR model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        epoch: Current epoch
        args: Training arguments
        global_step: Global step counter
    
    Returns:
        Average loss and updated global step
    """
    model.train()
    losses = []
    
    # Use distributed sampler if needed
    if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    for step, (view1, view2) in enumerate(train_loader):
        # Move data to device
        view1, view2 = view1.to(device), view2.to(device)
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda', enabled=args.mixed_precision):
            # Get projections
            z1 = model(view1)
            z2 = model(view2)
            
            # Compute loss
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
        
        # Backward and optimize with gradient accumulation
        scaler.scale(loss).backward()
        
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
            # Unscale gradients for proper clipping/logging
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step, scaler update, scheduler step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # Record loss
            losses.append(loss.item() * args.gradient_accumulation_steps)
            
            global_step += 1
            
            # Log metrics to wandb
            if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
                lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "train/loss": loss.item() * args.gradient_accumulation_steps,
                    "train/learning_rate": lr,
                    "train/global_step": global_step,
                }, step=global_step)
            
            # Log progress
            if step % args.log_every_n_steps == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch}/{args.epochs} | "
                      f"Step: {step}/{len(train_loader)} | "
                      f"Loss: {loss.item():.5f} | "
                      f"LR: {lr:.6f}")
    
    # Calculate average loss for the epoch
    avg_loss = sum(losses) / len(losses) if losses else 0
    
    # Log epoch metrics to wandb
    if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.log({
            "train/epoch": epoch,
            "train/avg_loss": avg_loss,
        }, step=global_step)
    
    return avg_loss, global_step


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, 
                    avg_loss, args, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: SimCLR model
        optimizer: Optimizer
        scheduler: Scheduler
        scaler: Gradient scaler
        epoch: Current epoch
        global_step: Current global step
        avg_loss: Average loss for this epoch
        args: Training arguments
        is_best: Whether this is the best model so far
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    rank = 0
    if hasattr(model, 'module'):  # Distributed training
        state_dict = model.module.state_dict()
        rank = dist.get_rank() if dist.is_initialized() else 0
    else:
        state_dict = model.state_dict()
    
    # Only save on the main process
    if rank == 0:
        checkpoint = {
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'global_step': global_step,
            'avg_loss': avg_loss,
            'args': vars(args)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save epoch checkpoint
        if epoch % args.save_every_n_epochs == 0:
            epoch_checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint-epoch-{epoch}.pt')
            torch.save(checkpoint, epoch_checkpoint_path)
            
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-best.pt')
            torch.save(checkpoint, best_checkpoint_path)
            
        print(f"Saved checkpoint at epoch {epoch}")
        
        # Log checkpoint to wandb
        if args.use_wandb and epoch % args.save_every_n_epochs == 0:
            wandb.save(epoch_checkpoint_path)
            if is_best:
                wandb.save(best_checkpoint_path)


def main_worker(gpu, args):
    """Main worker function for distributed training."""
    # If using multi-GPU with DDP, set up process group
    if args.distributed:
        rank = args.rank * args.gpus_per_node + gpu
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank
        )
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb on rank 0 only
    if args.use_wandb and (not args.distributed or rank == 0):
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"simclr3d-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
            settings=wandb.Settings(start_method="fork")
        )
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load bboxes data
    if rank == 0:
        print(f"Loading bbox data from {args.data_dir}...")
        
    loader = BBoxLoader(
        data_dir=args.data_dir,
        preproc_dir=args.preproc_dir,
        create_h5=True
    )
    volumes = loader.process_all_bboxes()
    
    if rank == 0:
        print(f"Loaded {len(volumes)} bbox volumes")
        for name, vol in volumes.items():
            print(f"  {name}: shape {vol.shape}")
    
    # Create dataset and dataloader
    dataset = RandomCubeDataset(
        bbox_volumes=volumes,
        cubes_per_bbox=args.cubes_per_bbox,
        cube_size=args.cube_size,
        mask_aware=args.mask_aware
    )
    
    # Use distributed sampler if needed
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=rank
        )
    else:
        sampler = None
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Calculate effective batch size
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    if args.distributed:
        effective_batch *= args.world_size
    
    # Adjust learning rate based on batch size
    lr = args.base_lr * (effective_batch / 256)
    
    if rank == 0:
        print(f"Using learning rate: {lr}")
        print(f"Effective batch size: {effective_batch}")
    
    # Create model
    model = SimCLR(
        backbone_type=args.backbone,
        in_channels=3,  # 3 channels for our synapse data
        hidden_dim=args.projector_hidden_dim,
        out_dim=args.projector_output_dim
    )
    
    # Move model to device
    model = model.to(device)
    
    # Convert BatchNorm to SyncBatchNorm for distributed training
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
    
    # Setup optimizer, scheduler, and scaler
    steps_per_epoch = len(train_loader)
    optimizer, scheduler, scaler = get_optimizer_and_scheduler(
        model,
        lr=lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if checkpoint['scaler_state_dict'] and args.mixed_precision:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        
        if 'avg_loss' in checkpoint:
            best_loss = checkpoint['avg_loss']
    
    # Log model summary to wandb
    if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.watch(model, log="all", log_freq=100)
    
    # Training loop
    if rank == 0:
        print(f"Starting training for {args.epochs} epochs")
        
    for epoch in range(start_epoch, args.epochs):
        # Switch to mask-aware sampling if configured
        if args.mask_aware_after_epoch is not None and epoch >= args.mask_aware_after_epoch:
            dataset.mask_aware = True
            if rank == 0:
                print(f"Switched to mask-aware sampling at epoch {epoch}")
                if args.use_wandb:
                    wandb.log({"train/mask_aware": True}, step=global_step)
        
        # Train for one epoch
        start_time = time.time()
        avg_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, args, global_step
        )
        epoch_time = time.time() - start_time
        
        if rank == 0:
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                  f"Avg Loss: {avg_loss:.5f}")
            
            # Save checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, avg_loss, args, is_best
            )
    
    # Save final model
    if rank == 0:
        print("Training completed, saving final model")
        
        # Save SimCLR model
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), 
                      os.path.join(args.checkpoint_dir, "simclr_final.pt"))
        else:
            torch.save(model.state_dict(), 
                      os.path.join(args.checkpoint_dir, "simclr_final.pt"))
        
        # Save backbone only (for downstream tasks)
        if isinstance(model, DDP):
            torch.save(model.module.backbone.state_dict(), 
                      os.path.join(args.checkpoint_dir, "backbone_final.pt"))
        else:
            torch.save(model.backbone.state_dict(), 
                      os.path.join(args.checkpoint_dir, "backbone_final.pt"))
        
        # Log final model to wandb
        if args.use_wandb:
            wandb.save(os.path.join(args.checkpoint_dir, "simclr_final.pt"))
            wandb.save(os.path.join(args.checkpoint_dir, "backbone_final.pt"))
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train SimCLR on synapse data')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing raw data')
    parser.add_argument('--preproc_dir', type=str, default='preproc',
                        help='Directory to store preprocessed data')
    parser.add_argument('--cube_size', type=int, default=80,
                        help='Size of cube to sample (default: 80)')
    parser.add_argument('--cubes_per_bbox', type=int, default=10000,
                        help='Number of cubes to sample per bbox (default: 10000)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU (default: 64)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='Base learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay (default: 1e-6)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='NT-Xent loss temperature (default: 0.1)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients (default: 1)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet3d',
                        choices=['resnet3d', 'swin3d'],
                        help='Backbone model (default: resnet3d)')
    parser.add_argument('--projector_hidden_dim', type=int, default=2048,
                        help='Projection MLP hidden dimension (default: 2048)')
    parser.add_argument('--projector_output_dim', type=int, default=256,
                        help='Projection MLP output dimension (default: 256)')
    
    # Mask-aware parameters
    parser.add_argument('--mask_aware', action='store_true',
                        help='Use mask-aware positive sampling')
    parser.add_argument('--mask_aware_after_epoch', type=int, default=None,
                        help='Switch to mask-aware sampling after this epoch')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every_n_epochs', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                        help='Log every N steps (default: 10)')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='simclr-3d-synapse',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    
    # Distributed training parameters
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0,
                        help='Process rank')
    parser.add_argument('--gpus_per_node', type=int, default=1,
                        help='Number of GPUs per node')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    
    args = parser.parse_args()
    
    # Set up distributed training
    if args.distributed:
        # Calculate world size
        args.world_size = args.gpus_per_node * args.world_size
        
        # Spawn processes
        mp.spawn(main_worker, nprocs=args.gpus_per_node, args=(args,))
    else:
        # Single-process training
        main_worker(0, args)


if __name__ == '__main__':
    main() 