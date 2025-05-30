import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint

class PatchEmbed3D(nn.Module):
    """
    3D Image to Patch Embedding
    """
    def __init__(self, patch_size=(4, 4, 4), in_channels=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Create projection layer
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # Input: B, C, D, H, W
        B, C, D, H, W = x.shape
        
        # Ensure dimensions are divisible by patch size
        pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            print(f"Padded input from {(B, C, D, H, W)} to {x.shape}")
        
        # Project patches
        x = self.proj(x)  # B, embed_dim, D', H', W'
        
        return x

class AliSwin3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: tuple = (4, 4, 4),
        window_size: tuple = (4, 8, 8),
        embed_dim: int = 96,
        depths: list = (2, 2, 6, 2),
        num_heads: list = (3, 6, 12, 24),
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        # Save parameters
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Create patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Create a simple feature extractor
        # This is a simplified version that focuses on extracting features
        # rather than classification
        self.features = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(embed_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(embed_dim * 2, embed_dim * 4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(embed_dim * 4),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(embed_dim * 4, embed_dim * 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(embed_dim * 8),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1)
        )
        
        # Set output dimension
        self.out_dim = embed_dim * 8
        print(f"Feature dimension: {self.out_dim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process input through patch embedding
        # print(f"Input shape: {x.shape}")
        
        # Create patch embeddings
        x = self.patch_embed(x)
        # print(f"After patch embedding: {x.shape}")
        
        # Extract features
        features = self.features(x)
        # print(f"Final features shape: {features.shape}")
        
        return features
