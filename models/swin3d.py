import torch
import torch.nn as nn
from torchvision.models.video import SwinTransformer3d
from torch.utils.checkpoint import checkpoint

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
        self.backbone = SwinTransformer3d(
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            in_chans=in_channels,
        )
        self.out_dim = embed_dim * 8
        
        self.use_checkpoint = use_checkpoint
        if use_checkpoint:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for Swin Transformer blocks to reduce memory usage."""
        for stage in [self.backbone.features.stage1, 
                      self.backbone.features.stage2, 
                      self.backbone.features.stage3, 
                      self.backbone.features.stage4]:
            for block in stage.blocks:
                block.forward = lambda x, block=block: checkpoint(block._forward, x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
