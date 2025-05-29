import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Tuple, Optional, Union
from models.swin3d import AliSwin3D            # NEW – real 3-D Swin


class SwinTransformer3D(nn.Module):
    """
    Adapts a 2D Swin Transformer to work with 3D volumes by applying 
    a pseudo-3D approach (process each slice, then aggregate features).
    """
    
    def __init__(self, in_channels: int = 5, embed_dim: int = 96, 
                 depths: List[int] = [2, 2, 6, 2], drop_rate: float = 0.0):
        """
        Initialize a 3D-adapted Swin Transformer.
        
        Args:
            in_channels: Number of input channels (5 for our synapse data)
            embed_dim: Initial embedding dimension
            depths: Number of transformer blocks in each stage
            drop_rate: Dropout rate
        """
        super().__init__()
        
        # Create 2D Swin-Tiny from timm with our custom input channels
        self.swin_model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            in_chans=in_channels,
            pretrained=False
        )
        
        # Get output dimension of the backbone
        self.out_dim = self.swin_model.head.in_features
        
        # Remove the classifier head
        self.swin_model.head = nn.Identity()
        
        # Add 3D pooling
        self.volume_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D-adapted Swin Transformer.
        
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
            
        Returns:
            Embedding tensor of shape [batch, out_dim]
        """
        batch_size, channels, depth, height, width = x.shape
        
        # Process each depth slice and collect features
        features_3d = []
        for d in range(depth):
            # Extract slice [batch, channels, height, width]
            x_slice = x[:, :, d, :, :]
            
            # Pass through 2D backbone
            features = self.swin_model(x_slice)  # [batch, out_dim]
            features_3d.append(features)
        
        # Stack along a new dimension [batch, depth, out_dim]
        features_3d = torch.stack(features_3d, dim=1)
        
        # Reshape to [batch, out_dim, depth, 1, 1] for 3D pooling
        features_3d = features_3d.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        
        # Apply 3D pooling
        pooled_features = self.volume_pool(features_3d).squeeze(-1).squeeze(-1).squeeze(-1)
        
        return pooled_features


class ResNet3D(nn.Module):
    """
    Basic 3D ResNet backbone for SimCLR.
    Simpler alternative to Swin Transformer that is easier to train.
    """
    
    def __init__(self, in_channels: int = 5):
        """
        Initialize a 3D ResNet18 model.
        
        Args:
            in_channels: Number of input channels (5 for our synapse data)
        """
        super().__init__()
        
        # Use torchvision's 3D ResNet implementation or similar
        try:
            import torchvision.models.video as video_models
            self.model = video_models.r3d_18(pretrained=False)
            
            # Modify first conv to accept our number of channels
            self.model.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Get output dimension
            self.out_dim = self.model.fc.in_features
            
            # Remove final classifier layer
            self.model.fc = nn.Identity()
            
        except (ImportError, AttributeError):
            # Fallback to a simplified 3D CNN if torchvision is not available
            self.model = nn.Sequential(
                nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                
                # Simple ResNet-like blocks
                self._make_layer(64, 64, 2),
                self._make_layer(64, 128, 2, stride=2),
                self._make_layer(128, 256, 2, stride=2),
                self._make_layer(256, 512, 2, stride=2),
                
                nn.AdaptiveAvgPool3d((1, 1, 1))
            )
            self.out_dim = 512
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Helper for creating a stack of residual blocks."""
        layers = []
        
        # Downsample if needed (changing dimensions or stride)
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # First block handles stride and downsampling
        layers.append(self._residual_block(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride=1, downsample=None):
        """Basic 3D residual block."""
        return nn.Sequential(
            # Layer 1
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            
            # Skip connection
            nn.Sequential(downsample) if downsample else nn.Identity(),
            
            # Final activation
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input: [batch, channels, depth, height, width]
        features = self.model(x)
        
        # Output: [batch, out_dim]
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        return features


class ProjectionMLP(nn.Module):
    """
    Projection MLP for SimCLR as described in Chen et al., 2020.
    
    Architecture: Linear(in_dim, hidden_dim) -> BN -> ReLU -> 
                 Linear(hidden_dim, hidden_dim) -> BN -> ReLU ->
                 Linear(hidden_dim, out_dim)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        """
        Initialize projection head.
        
        Args:
            in_dim: Input dimension (from backbone)
            hidden_dim: Hidden dimension
            out_dim: Output embedding dimension
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
        """Forward pass through projection head."""
        # Regular forward pass
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SimCLR(nn.Module):
    """
    SimCLR model combining backbone and projection head.
    
    The backbone extracts features from 3D volumes, while the projection head
    maps these features to a lower-dimensional space for contrastive learning.
    """
    
    def __init__(self, 
                 backbone_type: str = 'resnet3d',
                 in_channels: int = 3, 
                 hidden_dim: int = 2048, 
                 out_dim: int = 256):
        """
        Initialize SimCLR model.
        
        Args:
            backbone_type: Type of backbone ('resnet3d' or 'swin3d')
            in_channels: Number of input channels (default: 3 for our synapse data)
            hidden_dim: Hidden dimension for projection MLP
            out_dim: Output dimension for projection MLP
        """
        super().__init__()
        
        # Try to create the requested backbone, with fallback to ResNet3D
        if backbone_type == 'swin3d':
            print(f"Initializing Swin3D backbone with in_channels={in_channels}")
            self.backbone = AliSwin3D(
                in_channels=in_channels,
                patch_size=(4, 4, 4),
                window_size=(4, 8, 8),
                embed_dim=96,
                depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 24),
                drop_path_rate=0.1,
                use_checkpoint=True
            )
            print("Successfully initialized Swin3D backbone")
 
        else:
            self.backbone = ResNet3D(in_channels=in_channels)
        
        # Get backbone output dimension
        backbone_out_dim = self.backbone.out_dim
        
        # Create projection head
        self.projector = ProjectionMLP(
            in_dim=backbone_out_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone and projection head.
        
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
            
        Returns:
            Projected embedding of shape [batch, out_dim]
        """
        # Get features from backbone
        features = self.backbone(x)
        
        # Apply projector
        projections = self.projector(features)
        return projections
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from backbone only (for downstream tasks).
        
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
            
        Returns:
            Features of shape [batch, backbone_out_dim]
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper.
    Efficient vectorized implementation without per-row loops.
    
    Args:
        z1: Projected features from first augmentation, shape [batch_size, proj_dim]
        z2: Projected features from second augmentation, shape [batch_size, proj_dim]
        temperature: Temperature parameter
        
    Returns:
        NT-Xent loss value
    """
    batch_size = z1.shape[0]
    device = z1.device
    
    # Normalize feature vectors
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    
    # Concatenate representations: [z1, z2]
    representations = torch.cat([z1, z2], dim=0)
    
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(representations, representations.t()) / temperature
    
    # Create labels identifying positive pairs
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels])
    
    # For numerical stability, subtract the max from each row
    # Exclude the diagonal when computing the max to avoid including self-similarity
    sim_matrix_no_diag = similarity_matrix.clone()
    mask_diag = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim_matrix_no_diag.masked_fill_(mask_diag, float('-inf'))
    row_max, _ = sim_matrix_no_diag.max(dim=1, keepdim=True)
    similarity_matrix = similarity_matrix - row_max
    
    # Create mask to exclude self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    
    # Remove self-similarities by setting them to a large negative value (-inf)
    # This ensures they contribute 0 after exp() in the denominator
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
    
    # The positive pair for ith sample is at position labels[i]
    positives = similarity_matrix[torch.arange(2 * batch_size, device=device), labels]
        
    # Compute log_prob: log(exp(pos) / (sum of exp over all pairs except self))
    # Now self-similarity is properly excluded from the denominator
    denominator = torch.logsumexp(similarity_matrix, dim=1)
    log_prob = positives - denominator
    
    # Calculate the final loss
    loss = -log_prob.mean()
    
    return loss


if __name__ == "__main__":
    # Example usage
    batch_size = 8
    cube_size = 80
    in_channels = 5
    
    # Create random input data
    x = torch.randn(batch_size, in_channels, cube_size, cube_size, cube_size)
    
    # Create SimCLR model
    model = SimCLR(backbone_type='resnet3d', in_channels=in_channels)
    print(model)
    
    # Forward pass
    z = model(x)
    print(f"Output shape: {z.shape}")
    
    # Get features
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    
    # Test loss function
    x2 = torch.randn(batch_size, in_channels, cube_size, cube_size, cube_size)
    z1 = model(x)
    z2 = model(x2)
    loss = nt_xent_loss(z1, z2)
    print(f"NT-Xent loss: {loss.item()}") 