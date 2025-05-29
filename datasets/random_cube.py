import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
import h5py
import os

class Augmenter:
    """
    Implements augmentations for 3D volumes with both intensity and structural channels.
    
    Two augmentation streams:
    - Stream A: global transformations (rotations, elastic, blur, intensity jitter)
    - Stream B: stream A + crop-resize + optional mask dropout
    """
    
    def __init__(self, stream: str = 'A', cube_size: int = 80, p_mask_drop: float = 0.0):
        """
        Initialize augmenter with specific stream configuration.
        
        Args:
            stream: 'A' for global transforms, 'B' for global + local + mask dropout
            cube_size: Size of output cube (stream B may vary it during crop-resize)
            p_mask_drop: Probability of dropping a structural channel (stream B only)
        """
        self.stream = stream
        self.cube_size = cube_size
        self.p_mask_drop = p_mask_drop
    
    def apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random 90-degree rotation in 3D."""
        # Pick a random axis and rotation multiple of 90 degrees
        k = torch.randint(0, 4, (1,)).item()  # 0, 1, 2, 3 rotations
        if k == 0:
            return x
        
        # Pick a dimension to rotate in
        dim = torch.randint(0, 3, (1,)).item()  # 0, 1, 2 dimension
        
        if dim == 0:  # Rotate in Z-Y plane (around X)
            x = torch.rot90(x, k=k, dims=(1, 2))
        elif dim == 1:  # Rotate in Z-X plane (around Y)
            x = torch.rot90(x, k=k, dims=(1, 3))
        else:  # Rotate in Y-X plane (around Z)
            x = torch.rot90(x, k=k, dims=(2, 3))
        
        return x
    
    def apply_intensity_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply intensity jitter only to the raw channel (channel 0)."""
        # Only apply to raw channel
        scale = 0.1 * torch.randn(1).item() + 1.0  # Scale factor around 1.0
        shift = 0.1 * torch.randn(1).item()  # Small shift
        
        # Split the channels
        raw_channel = x[0:1]  # Keep dim for concatenation
        structure_channels = x[1:]
        
        # Apply transform only to raw
        raw_channel = scale * raw_channel + shift
        
        # Re-combine channels
        return torch.cat([raw_channel, structure_channels], dim=0)
    
    def apply_gaussian_blur(self, x: torch.Tensor, sigma_range: Tuple[float, float] = (0.1, 1.0)) -> torch.Tensor:
        """Apply 3D Gaussian blur to raw channel only using separable 3D convolutions."""
        # Only apply to raw channel
        raw_channel = x[0:1]  # Keep dim
        structure_channels = x[1:]
        
        # Skip with 50% probability
        if torch.rand(1).item() < 0.5:
            return x
        
        # Sample sigma - limit maximum sigma to 1.0 to prevent excessive blurring
        # This helps avoid completely wiping out features like clefts
        sigma = torch.rand(1).item() * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        
        # For 3D separable Gaussian blur:
        # 1. Create 1D Gaussian kernels
        # 2. Apply them sequentially along each dimension
        
        # Create 1D Gaussian kernel
        kernel_size = int(2 * round(2.5 * sigma) + 1)  # Cover ±2.5 sigma
        kernel_size = max(3, kernel_size)  # At least 3
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Create 1D Gaussian kernel
        half_size = kernel_size // 2
        x_coord = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
        gaussian_1d = torch.exp(-0.5 * (x_coord / sigma).pow(2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()  # Normalize
        
        # Create separable 3D kernels (one for each dimension)
        kernel_d = gaussian_1d.view(1, 1, -1, 1, 1)
        kernel_h = gaussian_1d.view(1, 1, 1, -1, 1)
        kernel_w = gaussian_1d.view(1, 1, 1, 1, -1)
        
        # Add batch dimension to raw channel for convolution
        raw_batch = raw_channel.unsqueeze(0)  # [1, 1, D, H, W]
        
        # Apply separable convolutions
        # Move tensor to CPU before convolution operations for better performance
        raw_batch_cpu = raw_batch.cpu()
        
        # Apply along depth (D)
        padding_d = (0, 0, 0, 0, half_size, half_size)
        blurred = F.conv3d(
            F.pad(raw_batch_cpu, padding_d, mode='replicate'),
            kernel_d,
            padding=0
        )
        
        # Apply along height (H)
        padding_h = (0, 0, half_size, half_size, 0, 0)
        blurred = F.conv3d(
            F.pad(blurred, padding_h, mode='replicate'),
            kernel_h,
            padding=0
        )
        
        # Apply along width (W)
        padding_w = (half_size, half_size, 0, 0, 0, 0)
        blurred = F.conv3d(
            F.pad(blurred, padding_w, mode='replicate'),
            kernel_w,
            padding=0
        )
        
        # Move back to the same device as input and remove batch dimension
        blurred_raw = blurred.to(x.device).squeeze(0)
        
        # Combine channels back
        return torch.cat([blurred_raw, structure_channels], dim=0)
    
    def apply_mask_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop structural channel by setting to zero."""
        if torch.rand(1).item() >= self.p_mask_drop:
            return x  # Skip dropout
        
        # Since we only have one structural channel (cleft mask at index 2),
        # we'll drop it with a 50% probability
        if torch.rand(1).item() < 0.5:
            # Clone to avoid modifying the input
            result = x.clone()
            result[2] = 0  # Zero out the cleft mask
            return result
        
        return x
    
    def apply_crop_resize(self, x: torch.Tensor, min_size: int = 64, max_size: int = 80) -> torch.Tensor:
        """Random crop followed by resize back to cube_size."""
        # Pick random crop size between min_size and max_size
        crop_size = torch.randint(min_size, max_size + 1, (1,)).item()
        
        # Get current size (should be cube_size in all dimensions)
        _, d, h, w = x.shape
        
        # Calculate crop start positions
        d_start = torch.randint(0, d - crop_size + 1, (1,)).item()
        h_start = torch.randint(0, h - crop_size + 1, (1,)).item()
        w_start = torch.randint(0, w - crop_size + 1, (1,)).item()
        
        # Crop the volume
        cropped = x[:, d_start:d_start+crop_size, h_start:h_start+crop_size, w_start:w_start+crop_size]
        
        # Resize back to cube_size using trilinear interpolation for raw channel
        # and nearest for mask channels
        raw_channel = cropped[0:1]
        structure_channels = cropped[1:]
        
        # Resize raw with trilinear
        raw_resized = F.interpolate(
            raw_channel.unsqueeze(0),  # Add batch dim
            size=(self.cube_size, self.cube_size, self.cube_size),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
        
        # Resize structure channels with nearest
        structure_resized = F.interpolate(
            structure_channels.unsqueeze(0),  # Add batch dim
            size=(self.cube_size, self.cube_size, self.cube_size),
            mode='nearest'
        ).squeeze(0)
        
        # Combine channels back
        return torch.cat([raw_resized, structure_resized], dim=0)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation stream to the input tensor."""
        # Global transformations (Stream A)
        x = self.apply_rotation(x)
        x = self.apply_intensity_jitter(x)
        x = self.apply_gaussian_blur(x)
        
        # Local transformations (Stream B only)
        if self.stream == 'B':
            x = self.apply_crop_resize(x)
            x = self.apply_mask_dropout(x)
        
        return x


class RandomCubeDataset(Dataset):
    """
    Dataset that samples random cubes from 3-channel bboxes for contrastive learning.
    
    Each item is a pair of differently augmented views of the same cube.
    """
    
    def __init__(self, 
                 bbox_volumes: Dict[str, torch.Tensor] = None,
                 data_dir: str = 'preproc',
                 bbox_names: List[str] = None,
                 cubes_per_bbox: int = 10000, 
                 cube_size: int = 80,
                 mask_aware: bool = False,
                 cleft_channel: int = 2):
        """
        Initialize the dataset.
        
        Args:
            bbox_volumes: Pre-loaded dict of bbox tensors (optional)
            data_dir: Directory with preprocessed H5 files (if bbox_volumes not provided)
            bbox_names: List of bbox names to use (if None, use all available)
            cubes_per_bbox: Number of random cubes to generate per bbox
            cube_size: Size of the cube to sample (default: 80³)
            mask_aware: Whether to ensure cleft overlap in positive pairs
            cleft_channel: Channel index containing cleft masks (default: 2)
        """
        self.cubes_per_bbox = cubes_per_bbox
        self.cube_size = cube_size
        self.mask_aware = mask_aware
        self.cleft_channel = cleft_channel
        
        # Load data either from provided volumes or from H5 files
        self.volumes = {}
        self.cleft_components = {}
        
        if bbox_volumes is not None:
            self.volumes = bbox_volumes
            self.bbox_names = list(bbox_volumes.keys())
        else:
            if bbox_names is None:
                # Look for all H5 files in the data directory
                h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
                self.bbox_names = [os.path.splitext(f)[0] for f in h5_files]
            else:
                self.bbox_names = bbox_names
            
            # Load all volumes from H5 files
            for bbox_name in self.bbox_names:
                h5_path = os.path.join(data_dir, f"{bbox_name}.h5")
                if os.path.exists(h5_path):
                    try:
                        with h5py.File(h5_path, 'r') as f:
                            self.volumes[bbox_name] = torch.from_numpy(f['volume'][()])
                    except Exception as e:
                        print(f"Error loading {h5_path}: {e}")
        
        if not self.volumes:
            raise ValueError(
                "No volumes loaded. Either provide bbox_volumes or ensure H5 files exist in data_dir."
            )
        
        # Pre-compute cleft component masks for mask-aware sampling
        if mask_aware:
            self._compute_cleft_components()
        
        # Create augmenters for the two views
        self.augmenter_A = Augmenter(stream='A', cube_size=cube_size)
        self.augmenter_B = Augmenter(stream='B', cube_size=cube_size, p_mask_drop=0.3)
        
        print(f"RandomCubeDataset initialized with {len(self.volumes)} bboxes, "
              f"{len(self)} total samples, mask_aware={mask_aware}")
    
    def _compute_cleft_components(self):
        """Pre-compute connected components for each cleft channel for mask-aware sampling."""
        from scipy import ndimage
        
        for bbox_name, volume in self.volumes.items():
            # Extract cleft mask
            cleft_mask = volume[self.cleft_channel].cpu().numpy()
            
            # Find connected components
            labeled_mask, num_features = ndimage.label(cleft_mask)
            
            # Store the labeled components
            self.cleft_components[bbox_name] = torch.from_numpy(labeled_mask)
            
            print(f"Found {num_features} cleft components in {bbox_name}")
    
    def _get_random_bbox(self) -> str:
        """Randomly select a bbox name."""
        idx = torch.randint(0, len(self.bbox_names), (1,)).item()
        return self.bbox_names[idx]
    
    def _sample_valid_cube(self, bbox_name: str) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Sample a random cube from the given bbox.
        
        If mask_aware is True, also returns the cleft component ID for this cube.
        """
        volume = self.volumes[bbox_name]
        _, d, h, w = volume.shape
        
        # Calculate valid ranges for the cube start position
        d_max = d - self.cube_size
        h_max = h - self.cube_size
        w_max = w - self.cube_size
        
        if d_max <= 0 or h_max <= 0 or w_max <= 0:
            raise ValueError(f"Cube size {self.cube_size} is too large for volume of shape {volume.shape}")
        
        # Random starting positions
        d_start = torch.randint(0, d_max + 1, (1,)).item()
        h_start = torch.randint(0, h_max + 1, (1,)).item()
        w_start = torch.randint(0, w_max + 1, (1,)).item()
        
        # Extract the cube
        cube = volume[:, d_start:d_start+self.cube_size, 
                     h_start:h_start+self.cube_size, 
                     w_start:w_start+self.cube_size]
        
        # For mask-aware sampling, return the cleft component ID
        cleft_component_id = None
        if self.mask_aware:
            cleft_components = self.cleft_components[bbox_name]
            # Extract the same region from the component map
            comp_region = cleft_components[d_start:d_start+self.cube_size, 
                                          h_start:h_start+self.cube_size, 
                                          w_start:w_start+self.cube_size]
            
            # Count unique components (excluding 0 which is background)
            unique_comps = torch.unique(comp_region)
            unique_comps = unique_comps[unique_comps > 0]  # Exclude background
            
            if len(unique_comps) > 0:
                # Choose the most frequently occurring component
                comp_counts = [(comp_region == comp).sum() for comp in unique_comps]
                cleft_component_id = unique_comps[np.argmax(comp_counts)].item()
        
        return cube, cleft_component_id
    
    def _check_component_overlap(self, view: torch.Tensor, 
                               orig_comp_id: int, bbox_name: str,
                               threshold: float = 0.4) -> bool:
        """
        Check if the augmented view still has sufficient overlap with the original component.
        
        Args:
            view: Augmented view tensor
            orig_comp_id: Original cleft component ID
            bbox_name: Name of the bbox for component lookup
            threshold: Minimum fraction of original component that must be present
            
        Returns:
            True if sufficient overlap, False otherwise
        """
        if orig_comp_id is None:
            return False
            
        # The cleft mask is in channel 2
        cleft_mask = view[self.cleft_channel]
        
        # Calculate cleft volume in the augmented view
        cleft_volume_in_view = cleft_mask.sum().item()
        
        # Check minimum meaningful size
        if cleft_volume_in_view < 10:  # Too small to be meaningful
            return False
            
        # Get the original cube's cleft mask
        # This is important to calculate what fraction of the *original* cleft is preserved
        # rather than just comparing to the entire cube volume
        original_cleft_mask = self._get_original_cleft_volume(bbox_name, orig_comp_id)
        
        # If we couldn't get the original volume (shouldn't happen), fall back to cube size check
        if original_cleft_mask is None or original_cleft_mask == 0:
            return cleft_volume_in_view > 100  # Require at least 100 voxels
        
        # Calculate what fraction of the original cleft volume is preserved
        preservation_ratio = cleft_volume_in_view / original_cleft_mask
        
        # Check if enough of the original component is preserved
        return preservation_ratio > threshold

    def _get_original_cleft_volume(self, bbox_name: str, comp_id: int) -> float:
        """
        Get the volume (voxel count) of the original cleft component.
        
        Args:
            bbox_name: Name of the bbox
            comp_id: Component ID to measure
            
        Returns:
            Volume (number of voxels) of the component
        """
        try:
            # Get the component map for this bbox
            comp_map = self.cleft_components[bbox_name]
            
            # Count voxels in this component
            component_volume = (comp_map == comp_id).sum().item()
            
            return component_volume
        except Exception as e:
            print(f"Error calculating component volume: {e}")
            return 0
    
    def _get_mask_aware_pair(self, max_attempts: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a pair of views with guaranteed cleft component overlap.
        
        Args:
            max_attempts: Maximum number of sampling attempts
            
        Returns:
            Tuple of two augmented views
        """
        for _ in range(max_attempts):
            bbox_name = self._get_random_bbox()
            cube, comp_id = self._sample_valid_cube(bbox_name)
            
            # Skip if no significant cleft component found
            if comp_id is None or cube[self.cleft_channel].sum() < 100:
                continue
                
            # Apply augmentations
            view_a = self.augmenter_A(cube)
            view_b = self.augmenter_B(cube)
            
            # Check if both views retain sufficient cleft component
            if cube[self.cleft_channel].sum() > 0 and \
               view_a[self.cleft_channel].sum() > 0 and \
               view_b[self.cleft_channel].sum() > 0:
                return view_a, view_b
                
        # If we reach here, just return regular augmentations
        bbox_name = self._get_random_bbox()
        cube, _ = self._sample_valid_cube(bbox_name)
        return self.augmenter_A(cube), self.augmenter_B(cube)
    
    def __len__(self) -> int:
        """Total number of samples in the dataset."""
        return len(self.volumes) * self.cubes_per_bbox
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of differently augmented views of the same cube.
        
        Args:
            idx: Index (not used except for distributing across workers)
            
        Returns:
            Tuple of two tensors, each of shape (3, cube_size, cube_size, cube_size)
        """
        if self.mask_aware:
            view_a, view_b = self._get_mask_aware_pair()
        else:
            # Regular sampling (not mask-aware)
            bbox_name = self._get_random_bbox()
            cube, _ = self._sample_valid_cube(bbox_name)
            
            # Apply different augmentations to get two views
            view_a = self.augmenter_A(cube)
            view_b = self.augmenter_B(cube)
        
        # Ensure tensors are in float32 format for mixed precision training
        view_a = view_a.float()
        view_b = view_b.float()
        
        return view_a, view_b


if __name__ == "__main__":
    # Example usage
    from bbox_loader import BBoxLoader
    
    # First load and preprocess the bboxes
    loader = BBoxLoader(data_dir='data', preproc_dir='preproc', create_h5=True)
    volumes = loader.process_all_bboxes()
    
    # Create the dataset
    dataset = RandomCubeDataset(
        bbox_volumes=volumes,
        cubes_per_bbox=100,
        cube_size=80,
        mask_aware=True
    )
    
    # Fetch a sample
    view_a, view_b = dataset[0]
    print(f"View A shape: {view_a.shape}")
    print(f"View B shape: {view_b.shape}")
    
    # Check that they are different augmentations of the same content
    print(f"Raw channel correlation: {torch.corrcoef(view_a[0].flatten(), view_b[0].flatten())[0,1]}") 