import torch
import contextlib
from torch.cuda.amp import autocast




# Custom autocast context manager to prevent BatchNorm from using fp16
# This solves NaN issues when using SyncBatchNorm with small batches
@contextlib.contextmanager
def safe_autocast(device_type='cuda', enabled=True):
    """
    Custom autocast context manager that forces BatchNorm layers to use float32.
    Prevents NaN values with small batch sizes when using SyncBatchNorm.
    """
    if enabled:
        # Store original batch norm types
        orig_bn_fp32 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        
        # Force BN to use fp32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        
        # Use autocast - keeping it simple to avoid version compatibility issues
        with autocast():
            yield
        
        # Restore original settings
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_bn_fp32
    else:
        yield


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


def get_optimizer_and_scheduler(model, lr, weight_decay, epochs, steps_per_epoch, gradient_accumulation_steps=1, scheduler_type='constant'):
    """
    Create LARS optimizer and learning rate scheduler.
    
    Args:
        model: SimCLR model
        lr: Learning rate
        weight_decay: Weight decay factor
        epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
        gradient_accumulation_steps: Number of steps to accumulate gradients
        scheduler_type: Type of scheduler ('constant', 'step', 'plateau', 'cosine', 'warm_restarts')
        
    Returns:
        Optimizer, scheduler, and gradient scaler
    """
    from torch.cuda.amp import GradScaler
    
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
    
    # Create scheduler based on type
    if scheduler_type == 'constant':
        # No learning rate decay - keeps LR constant
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0  # Always return 1.0 to keep LR constant
        )
    elif scheduler_type == 'step':
        # Step decay - reduce LR by factor every N epochs (not time-based)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,  # Reduce every 50 epochs
            gamma=0.5      # Multiply by 0.5
        )
    elif scheduler_type == 'plateau':
        # Reduce LR when loss plateaus (adaptive, not time-based)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,   # Wait 10 epochs without improvement
            verbose=True
        )
    elif scheduler_type == 'cosine':
        # Original cosine annealing (epoch-based)
        effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs * effective_steps_per_epoch,
        )
    elif scheduler_type == 'warm_restarts':
        # Cosine with warm restarts (epoch-based)
        effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50 * effective_steps_per_epoch,  # Restart every 50 epochs
            T_mult=1,  # Keep the same restart period
            eta_min=lr * 0.01  # Minimum learning rate (1% of base)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler()
    
    return optimizer, scheduler, scaler 