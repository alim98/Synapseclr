#!/usr/bin/env python3
"""
Train SimCLR on individual synapse cubes extracted from Excel annotation data.

This script uses the SynapseLoader to extract specific synapses based on 
coordinates from Excel files, following the approach described in the user's 
reference code.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from models.simclr3d import SimCLR, nt_xent_loss
from datasets.synapse_loader import SynapseLoader
from datasets.synapse_dataset import SynapseDataset
from utils.training_utils import get_optimizer_and_scheduler, safe_autocast

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        print('Not using distributed mode')
        return False, 0, 0, 0

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu)
    dist.barrier()
    return True, rank, world_size, gpu

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, 
                epoch, args, global_step=0, writer=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (view1, view2) in enumerate(train_loader):
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        
        # Forward pass
        with safe_autocast(enabled=args.mixed_precision):
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        if args.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            if args.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        # Logging
        total_loss += loss.item() * args.gradient_accumulation_steps
        num_batches += 1
        
        if batch_idx % args.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            if args.distributed:
                rank = dist.get_rank()
                if rank == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item() * args.gradient_accumulation_steps:.5f}, '
                          f'LR: {lr:.6f}')
            else:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item() * args.gradient_accumulation_steps:.5f}, '
                      f'LR: {lr:.6f}')
        
        # TensorBoard logging
        if writer and global_step % args.log_interval == 0:
            actual_loss = loss.item() * args.gradient_accumulation_steps
            writer.add_scalar('Loss/Train', actual_loss, global_step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            # Log additional metrics for monitoring
            writer.add_scalar('Loss/Batch_Loss', actual_loss, global_step)
            writer.add_scalar('Training/Epoch', epoch, global_step)
            writer.add_scalar('Training/Progress', global_step / (len(train_loader) * args.epochs), global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, 
                    avg_loss, args, is_best=False):
    """Save training checkpoint."""
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if args.mixed_precision else None,
        'epoch': epoch,
        'global_step': global_step,
        'avg_loss': avg_loss,
        'args': args
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
        print(f"New best model saved with loss: {avg_loss:.5f}")

def main_worker(gpu, args):
    """Main worker function."""
    # Setup device
    if args.distributed:
        rank = dist.get_rank()
        device = torch.device(f'cuda:{gpu}')
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"Using device: {device}")
    
    # Setup TensorBoard
    writer = None
    if rank == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.run_name}_{timestamp}" if args.run_name else f"synapse_simclr_{timestamp}"
        log_dir = os.path.join(args.tensorboard_dir, run_name)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        
        # Log hyperparameters
        hparams = {
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'temperature': args.temperature,
            'cube_size': args.cube_size,
            'epochs': args.epochs,
            'base_lr': args.base_lr,
        }
        writer.add_hparams(hparams, {})
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load synapse data
    if rank == 0:
        print(f"Loading synapse data from {args.data_dir}...")
        
    loader = SynapseLoader(
        data_dir=args.data_dir,
        preproc_dir=args.preproc_dir,
        cube_size=args.cube_size
    )
    
    # Process all synapses  
    synapses = loader.process_all_synapses(memory_efficient=args.memory_efficient)
    
    # If memory_efficient was used, load from paths
    if synapses and isinstance(next(iter(synapses.values())), str):
        if rank == 0:
            print("Loading synapses from H5 files...")
        synapses = loader.load_from_paths(synapses)
        
        # Verify all synapses loaded successfully
        failed_loads = [name for name, syn in synapses.items() if syn is None]
        if failed_loads:
            raise RuntimeError(f"Failed to load synapses: {failed_loads}")
    
    if rank == 0:
        print(f"Loaded {len(synapses)} synapse cubes")
        
        # Print some statistics
        bbox_counts = {}
        for synapse_id in synapses.keys():
            bbox_name = synapse_id.split('_')[0] + synapse_id.split('_')[1]  # e.g., bbox1
            bbox_counts[bbox_name] = bbox_counts.get(bbox_name, 0) + 1
        
        print("Synapse distribution per bbox:")
        for bbox_name, count in sorted(bbox_counts.items()):
            print(f"  {bbox_name}: {count} synapses")
    
    # Create dataset and dataloader
    dataset = SynapseDataset(
        synapse_cubes=synapses,
        augment=True
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
        steps_per_epoch=steps_per_epoch,
        gradient_accumulation_steps=args.gradient_accumulation_steps
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
    
    # Training loop
    if rank == 0:
        print(f"Starting training for {args.epochs} epochs")
        print(f"Training on {len(synapses)} individual synapse cubes")
        
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        start_time = time.time()
        avg_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, args, global_step, writer
        )
        epoch_time = time.time() - start_time
        
        if rank == 0:
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                  f"Avg Loss: {avg_loss:.5f}")
            
            # Log epoch-level metrics
            if writer:
                writer.add_scalar('Loss/Epoch_Avg', avg_loss, epoch)
                writer.add_scalar('Training/Epoch_Time', epoch_time, epoch)
                writer.add_scalar('Training/Best_Loss', best_loss, epoch)
                writer.add_scalar('Training/Current_LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            
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
                      os.path.join(args.checkpoint_dir, "synapse_simclr_final.pt"))
        else:
            torch.save(model.state_dict(), 
                      os.path.join(args.checkpoint_dir, "synapse_simclr_final.pt"))
        
        # Save backbone only (for downstream tasks)
        if isinstance(model, DDP):
            torch.save(model.module.backbone.state_dict(), 
                      os.path.join(args.checkpoint_dir, "synapse_backbone_final.pt"))
        else:
            torch.save(model.backbone.state_dict(), 
                      os.path.join(args.checkpoint_dir, "synapse_backbone_final.pt"))
        
        # Close tensorboard writer
        if writer:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train SimCLR on individual synapse cubes')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory containing raw, seg folders and Excel files')
    parser.add_argument('--preproc_dir', type=str, default='preproc_synapses',
                        help='Directory to store preprocessed synapse cubes')
    parser.add_argument('--cube_size', type=int, default=80,
                        help='Size of synapse cubes to extract')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--base_lr', type=float, default=0.0003,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature parameter for contrastive loss')
    
    # Model parameters
    parser.add_argument('--projector_hidden_dim', type=int, default=512,
                        help='Hidden dimension of projector')
    parser.add_argument('--projector_output_dim', type=int, default=128,
                        help='Output dimension of projector')
    
    # Optimization parameters
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--memory_efficient', action='store_true',
                        help='Use memory efficient loading (save synapses to H5 files)')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard_logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this training run')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    # Setup distributed training
    if args.distributed:
        distributed, rank, world_size, gpu = setup_distributed()
        args.world_size = world_size
        if distributed:
            main_worker(gpu, args)
        else:
            print("Failed to setup distributed training")
    else:
        main_worker(0, args)


if __name__ == '__main__':
    main() 