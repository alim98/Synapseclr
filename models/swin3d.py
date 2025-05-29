import torch
import torch.nn as nn
from torchvision.models.video import SwinTransformer3d
from torch.utils.checkpoint import checkpoint
import torchvision
import numpy as np

class AliSwin3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        patch_size: tuple = (4, 4, 4),
        window_size: tuple = (4, 8, 8),
        embed_dim: int = 96,
        depths: list = (2, 2, 6, 2),
        num_heads: list = (3, 6, 12, 24),
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        # Check torchvision version
        print(f"Using torchvision version: {torchvision.__version__}")
        
        # For torchvision >= 0.15.0, the parameter name is different
        # Print all available parameters for debugging
        import inspect
        params = inspect.signature(SwinTransformer3d.__init__).parameters
        param_names = list(params.keys())
        print(f"Available parameters for SwinTransformer3d: {param_names}")
        
        # Convert window_size and patch_size to numpy arrays if needed
        # This is to support the .copy() method used in the implementation
        window_size_param = np.array(window_size) if torchvision.__version__ >= '0.22.0' else window_size
        patch_size_param = np.array(patch_size) if torchvision.__version__ >= '0.22.0' else patch_size
        
        # Common parameters that should be available in all versions
        swin_kwargs = {
            'patch_size': patch_size_param,
            'embed_dim': embed_dim,
            'depths': depths,
            'num_heads': num_heads,
            'window_size': window_size_param,
            'num_classes': 0,
        }
        
        # Handle input channels based on available parameters
        # In newer versions of torchvision, the parameter might be different
        if 'in_channels' in param_names:
            swin_kwargs['in_channels'] = in_channels
        
        # Handle drop path rate / stochastic depth
        if 'stochastic_depth_prob' in param_names:
            swin_kwargs['stochastic_depth_prob'] = drop_path_rate
        elif 'drop_path_rate' in param_names:
            swin_kwargs['drop_path_rate'] = drop_path_rate
        
        # Print the final parameters we're using
        print(f"Initializing SwinTransformer3d with parameters: {swin_kwargs}")
        
        # Initialize the Swin Transformer
        self.backbone = SwinTransformer3d(**swin_kwargs)
        
        # Create a classifier head if the model doesn't include one
        # or if the classifier head isn't producing valid output
        if torchvision.__version__ >= '0.22.0':
            # For newer versions, ensure we have proper output
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(1),
                nn.Linear(embed_dim * 8, embed_dim * 8)
            )
        else:
            # For older versions, may not need a separate head
            self.head = nn.Identity()
            
        # Set output dimension
        self.out_dim = embed_dim * 8
        print("Successfully initialized SwinTransformer3d")
            
        # Enable gradient checkpointing if requested
        self.use_checkpoint = use_checkpoint
        if use_checkpoint:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for Swin Transformer blocks to reduce memory usage."""
        try:
            # First, check the structure of the backbone
            if hasattr(self.backbone, 'features'):
                # For newer torchvision versions
                stages = []
                for i in range(1, 5):
                    stage_name = f'stage{i}'
                    if hasattr(self.backbone.features, stage_name):
                        stages.append(getattr(self.backbone.features, stage_name))
                
                for stage in stages:
                    if hasattr(stage, 'blocks'):
                        for block in stage.blocks:
                            # Store the original forward method
                            if not hasattr(block, '_forward'):
                                block._forward = block.forward
                            # Apply checkpoint
                            block.forward = lambda x, block=block: checkpoint(block._forward, x)
                
                print("Gradient checkpointing enabled for features.stageX.blocks")
            
            elif hasattr(self.backbone, 'layers'):
                # For older versions with a different structure
                for layer in self.backbone.layers:
                    if hasattr(layer, 'blocks'):
                        for block in layer.blocks:
                            if not hasattr(block, '_forward'):
                                block._forward = block.forward
                            block.forward = lambda x, block=block: checkpoint(block._forward, x)
                
                print("Gradient checkpointing enabled for layers.blocks")
            
            else:
                # Try to find Transformer blocks in any location
                def apply_checkpoint_to_blocks(module):
                    if isinstance(module, nn.Module) and hasattr(module, 'forward'):
                        children = list(module.named_children())
                        if len(children) == 0 and 'block' in module.__class__.__name__.lower():
                            # Looks like a transformer block
                            if not hasattr(module, '_forward'):
                                module._forward = module.forward
                            module.forward = lambda x, mod=module: checkpoint(mod._forward, x)
                        else:
                            # Recursively apply to children
                            for _, child in children:
                                apply_checkpoint_to_blocks(child)
                
                apply_checkpoint_to_blocks(self.backbone)
                print("Applied gradient checkpointing to transformer blocks")
                
        except Exception as e:
            print(f"Warning: Could not enable gradient checkpointing: {e}")
            print("Training will continue but memory usage may be higher")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input dimensions are divisible by patch size
        if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'patch_size'):
            patch_size = self.backbone.patch_embed.patch_size
            if isinstance(patch_size, (tuple, list, np.ndarray)):
                patch_d, patch_h, patch_w = patch_size
            else:
                patch_d = patch_h = patch_w = patch_size
                
            # Get current dimensions
            _, _, d, h, w = x.shape
            
            # Ensure dimensions are divisible by patch size
            # Pad if necessary
            pad_d = (patch_d - (d % patch_d)) % patch_d
            pad_h = (patch_h - (h % patch_h)) % patch_h
            pad_w = (patch_w - (w % patch_w)) % patch_w
            
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                # Apply padding to make dimensions divisible by patch size
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        
        # Forward pass through backbone
        features = self.backbone(x)
        
        # Handle different backbone outputs
        # Add some checks for different torchvision versions
        if hasattr(self.backbone, 'head') and isinstance(self.backbone.head, nn.Identity):
            # If backbone has no classification head, the output might need processing
            if features.dim() > 2:
                # For newer versions (0.22.0+), use our head
                features = self.head(features)
        
        # Ensure we have the right shape output
        if features.dim() > 2:
            # If still not flattened, flatten it
            features = torch.flatten(features, 1)
            
        return features
