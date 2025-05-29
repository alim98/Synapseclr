import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import SwinTransformer3d
from typing import List, Tuple, Optional


class TrueSwin3D(nn.Module):
    """
    A true 3D Swin Transformer implementation for 80³ synapse cubes.
    Based on the torchvision.models.video.SwinTransformer3d implementation.
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        window_size: Tuple[int, int, int] = (4, 8, 8),
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        """
        Initialize a true 3D Swin Transformer.
        
        Args:
            in_channels: Number of input channels (5 for synapse data: raw + 4 masks)
            patch_size: Size of each patch (depth, height, width)
            window_size: Size of attention window (depth, height, width)
            embed_dim: Initial embedding dimension (C₀)
            depths: Number of blocks in each stage
            num_heads: Number of attention heads in each stage
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer
        """
        super().__init__()
        
        # Create the 3D Swin Transformer backbone using torchvision implementation
        self.backbone = SwinTransformer3d(
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            num_classes=0,  # removes classification head
            in_chans=in_channels
        )
        
        # Get output dimension (after final pooling)
        self.out_dim = embed_dim * 8  # embed_dim * 2^(num_stages-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D Swin Transformer.
        
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
            
        Returns:
            Feature tensor of shape [batch, out_dim]
        """
        # Input shape: [B, C, D, H, W] = [B, 5, 80, 80, 80]
        features = self.backbone(x)
        
        return features


class SimCLRProjectionHead(nn.Module):
    """
    Projection head for SimCLR as specified in the blueprint:
    - 768 → 2048 → 2048 → 256 with BN after each hidden layer
    - L2-normalization of the 256-D output
    """
    
    def __init__(self, in_dim: int = 768, hidden_dim: int = 2048, out_dim: int = 256):
        """
        Initialize SimCLR projection head.
        
        Args:
            in_dim: Input dimension (from backbone)
            hidden_dim: Hidden dimension (2048 as per spec)
            out_dim: Output dimension (256 as per spec)
        """
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.layer3 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.
        
        Args:
            x: Input tensor of shape [batch, in_dim]
            
        Returns:
            L2-normalized embeddings of shape [batch, out_dim]
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # L2-normalize the output
        x = F.normalize(x, p=2, dim=1)
        
        return x


class Swin3DSimCLR(nn.Module):
    """
    Complete SimCLR model with true 3D Swin Transformer backbone
    and projection head for contrastive learning.
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        window_size: Tuple[int, int, int] = (4, 8, 8),
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        hidden_dim: int = 2048,
        out_dim: int = 256,
        drop_path_rate: float = 0.1,
    ):
        """
        Initialize Swin3DSimCLR model.
        
        Args:
            in_channels: Number of input channels
            patch_size: Size of each patch
            window_size: Size of attention window
            embed_dim: Initial embedding dimension
            depths: Number of blocks in each stage
            num_heads: Number of attention heads in each stage
            hidden_dim: Hidden dimension in projection head
            out_dim: Output dimension for projection head
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        
        # Initialize the 3D Swin Transformer backbone
        self.backbone = TrueSwin3D(
            in_channels=in_channels,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate
        )
        
        # Initialize projection head for SimCLR
        self.projection = SimCLRProjectionHead(
            in_dim=self.backbone.out_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Swin3DSimCLR.
        
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
            
        Returns:
            L2-normalized embeddings of shape [batch, out_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project features
        embeddings = self.projection(features)
        
        return embeddings
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without projection head (for downstream tasks).
        
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
            
        Returns:
            Features of shape [batch, backbone.out_dim]
        """
        return self.backbone(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper.
    
    Args:
        z1: First set of embeddings [batch_size, dim]
        z2: Second set of embeddings [batch_size, dim]
        temperature: Temperature parameter
        
    Returns:
        NT-Xent loss
    """
    # Concatenate embeddings from both augmentations
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2*batch_size, dim]
    
    # Compute similarity matrix
    sim = torch.mm(z, z.T) / temperature  # [2*batch_size, 2*batch_size]
    
    # Create mask for positive pairs
    sim_i_j = torch.diag(sim, batch_size)  # Similarity between z_i and z_j
    sim_j_i = torch.diag(sim, -batch_size)  # Similarity between z_j and z_i
    
    # We need to mask the diagonal part of the similarity matrix
    # as they are trivial matches of samples with themselves
    mask = torch.ones_like(sim)
    mask = mask.fill_diagonal_(0)
    
    # Find negative similarities
    neg_sim = mask * torch.exp(sim)
    
    # Compute loss for both augmentation directions
    loss_i = -sim_i_j + torch.log(torch.sum(neg_sim[:batch_size], dim=1))
    loss_j = -sim_j_i + torch.log(torch.sum(neg_sim[batch_size:], dim=1))
    
    # Average over the batch
    loss = torch.mean(loss_i + loss_j) / 2
    
    return loss 