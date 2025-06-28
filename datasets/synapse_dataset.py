import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random

class SynapseDataset(Dataset):
    """
    Dataset for individual synapse cubes extracted from Excel annotations.
    
    This dataset takes synapse cubes from SynapseLoader and applies augmentations
    for contrastive learning. Each synapse is treated as an individual sample.
    """
    
    def __init__(self, 
                 synapse_cubes: Dict[str, torch.Tensor],
                 augment: bool = True):
        """
        Initialize the synapse dataset.
        
        Args:
            synapse_cubes: Dict mapping synapse_id -> 3-channel tensor
            augment: Whether to apply augmentations for contrastive learning
        """
        self.synapse_cubes = synapse_cubes
        self.synapse_ids = list(synapse_cubes.keys())
        self.augment = augment
        
        # Filter out None values
        valid_synapses = {k: v for k, v in synapse_cubes.items() if v is not None}
        self.synapse_cubes = valid_synapses
        self.synapse_ids = list(valid_synapses.keys())
        
        print(f"SynapseDataset initialized with {len(self.synapse_ids)} synapses")
        
    def __len__(self):
        return len(self.synapse_ids)
    
    def __getitem__(self, idx):
        """
        Get a synapse cube and apply augmentations.
        
        Returns:
            Tuple of (view1, view2) for contrastive learning if augment=True,
            otherwise returns the original cube.
        """
        synapse_id = self.synapse_ids[idx]
        cube = self.synapse_cubes[synapse_id]
        
        if not self.augment:
            return cube
        
        # Apply augmentations to create two views for contrastive learning
        view1 = self._augment_cube(cube)
        view2 = self._augment_cube(cube)
        
        return view1, view2
    
    def _augment_cube(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Apply STRONG augmentations to a synapse cube for better contrastive learning.
        
        Enhanced augmentations include:
        - Random flips along each axis
        - Stronger random rotations (not just 90 degrees)
        - Elastic deformations
        - More aggressive Gaussian noise
        - Stronger intensity variations
        - Random cropping and scaling
        - Gamma correction
        """
        # Clone the cube to avoid modifying the original
        augmented = cube.clone()
        
        # Random flips along each axis (higher probability)
        if random.random() > 0.3:  # Increased from 0.5
            augmented = torch.flip(augmented, dims=[1])  # flip z
        if random.random() > 0.3:
            augmented = torch.flip(augmented, dims=[2])  # flip y
        if random.random() > 0.3:
            augmented = torch.flip(augmented, dims=[3])  # flip x
        
        # STRONGER ROTATIONS: Not just 90-degree steps
        # Random rotation in xy plane (around z axis) - any angle
        if random.random() > 0.4:
            # For continuous rotation, we'll use discrete steps for simplicity
            k = random.randint(1, 7)  # More rotation options
            augmented = torch.rot90(augmented, k=k, dims=[2, 3])
        
        # Random rotation in xz plane (around y axis)
        if random.random() > 0.4:
            k = random.randint(1, 7)
            augmented = torch.rot90(augmented, k=k, dims=[1, 3])
        
        # Random rotation in yz plane (around x axis)
        if random.random() > 0.4:
            k = random.randint(1, 7)
            augmented = torch.rot90(augmented, k=k, dims=[1, 2])
        
        # ELASTIC DEFORMATION (simplified version using random scaling)
        if random.random() > 0.3:
            augmented = self._apply_elastic_deformation(augmented)
        
        # RANDOM CROPPING AND SCALING
        if random.random() > 0.4:
            augmented = self._random_crop_and_scale(augmented)
        
        # Apply stronger noise and intensity variations to the raw channel (channel 0)
        raw_channel = augmented[0]  # Raw EM intensity
        
        # STRONGER Gaussian noise
        if random.random() > 0.3:  # More frequent
            noise_std = random.uniform(0.02, 0.15)  # Increased from 0.01-0.05
            noise = torch.randn_like(raw_channel) * noise_std
            augmented[0] = torch.clamp(raw_channel + noise, 0, 1)
        
        # STRONGER intensity variations (contrast and brightness)
        if random.random() > 0.3:
            # Stronger contrast: wider range
            contrast_factor = random.uniform(0.5, 1.8)  # Increased from 0.8-1.2
            # Stronger brightness: wider range
            brightness_offset = random.uniform(-0.3, 0.3)  # Increased from -0.1, 0.1
            
            adjusted = raw_channel * contrast_factor + brightness_offset
            augmented[0] = torch.clamp(adjusted, 0, 1)
        
        # GAMMA CORRECTION for additional intensity variation
        if random.random() > 0.4:
            gamma = random.uniform(0.5, 2.0)
            augmented[0] = torch.pow(augmented[0], gamma)
        
        # SALT AND PEPPER NOISE
        if random.random() > 0.6:
            augmented = self._add_salt_pepper_noise(augmented)
        
        return augmented
    
    def _apply_elastic_deformation(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Apply elastic deformation by random local scaling.
        Simplified version - randomly scale different regions.
        """
        C, D, H, W = cube.shape
        
        # Create random scaling factors for different regions
        if random.random() > 0.5:
            # Random scaling in depth
            scale_factor = random.uniform(0.8, 1.2)
            new_depth = max(int(D * scale_factor), D // 2)
            
            # Interpolate along depth dimension
            indices = torch.linspace(0, D-1, new_depth)
            indices = indices.long().clamp(0, D-1)
            
            scaled = cube[:, indices, :, :]
            
            # Resize back to original depth if needed
            if new_depth != D:
                # Simple repetition/subsampling to get back to original size
                if new_depth < D:
                    # Repeat frames
                    repeat_factor = D // new_depth + 1
                    scaled = scaled.repeat(1, repeat_factor, 1, 1)[:, :D, :, :]
                else:
                    # Subsample
                    scaled = scaled[:, :D, :, :]
            
            cube = scaled
        
        return cube
    
    def _random_crop_and_scale(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Randomly crop a region and scale it back to original size.
        """
        C, D, H, W = cube.shape
        
        # Random crop size (between 70% and 95% of original)
        crop_ratio = random.uniform(0.7, 0.95)
        
        crop_d = max(int(D * crop_ratio), D // 2)
        crop_h = max(int(H * crop_ratio), H // 2)
        crop_w = max(int(W * crop_ratio), W // 2)
        
        # Random crop position
        start_d = random.randint(0, D - crop_d)
        start_h = random.randint(0, H - crop_h)
        start_w = random.randint(0, W - crop_w)
        
        # Crop
        cropped = cube[:, start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Scale back using nearest neighbor interpolation (simple repetition)
        # For EM data, we want to preserve discrete values
        scaled = torch.nn.functional.interpolate(
            cropped.unsqueeze(0), 
            size=(D, H, W), 
            mode='nearest'
        ).squeeze(0)
        
        return scaled
    
    def _add_salt_pepper_noise(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Add salt and pepper noise to the raw channel only.
        """
        raw_channel = cube[0].clone()
        
        # Salt and pepper noise
        noise_ratio = random.uniform(0.001, 0.01)  # Small amount
        
        # Salt noise (set random pixels to 1)
        salt_mask = torch.rand_like(raw_channel) < noise_ratio / 2
        raw_channel[salt_mask] = 1.0
        
        # Pepper noise (set random pixels to 0)
        pepper_mask = torch.rand_like(raw_channel) < noise_ratio / 2
        raw_channel[pepper_mask] = 0.0
        
        cube[0] = raw_channel
        return cube
    
    def get_synapse_info(self, idx: int) -> str:
        """Get synapse ID for debugging purposes."""
        return self.synapse_ids[idx]
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'num_synapses': len(self.synapse_ids),
            'cube_shapes': [],
            'bbox_distribution': {}
        }
        
        for synapse_id in self.synapse_ids:
            cube = self.synapse_cubes[synapse_id]
            stats['cube_shapes'].append(cube.shape)
            
            # Extract bbox name from synapse_id (e.g., "bbox1_synapse_name")
            bbox_name = synapse_id.split('_')[0] + synapse_id.split('_')[1]  # bbox1
            stats['bbox_distribution'][bbox_name] = stats['bbox_distribution'].get(bbox_name, 0) + 1
        
        return stats


class SynapseContrastiveDataset(Dataset):
    """
    Alternative dataset that samples positive pairs from the same synapse
    and negative pairs from different synapses.
    """
    
    def __init__(self, 
                 synapse_cubes: Dict[str, torch.Tensor],
                 samples_per_epoch: int = 1000):
        """
        Initialize the contrastive synapse dataset.
        
        Args:
            synapse_cubes: Dict mapping synapse_id -> 3-channel tensor
            samples_per_epoch: Number of positive pairs to generate per epoch
        """
        self.synapse_cubes = synapse_cubes
        self.synapse_ids = list(synapse_cubes.keys())
        self.samples_per_epoch = samples_per_epoch
        
        # Filter out None values
        valid_synapses = {k: v for k, v in synapse_cubes.items() if v is not None}
        self.synapse_cubes = valid_synapses
        self.synapse_ids = list(valid_synapses.keys())
        
        print(f"SynapseContrastiveDataset initialized with {len(self.synapse_ids)} synapses")
        print(f"Will generate {samples_per_epoch} positive pairs per epoch")
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        """
        Generate a positive pair by applying different augmentations to the same synapse.
        """
        # Randomly select a synapse
        synapse_id = random.choice(self.synapse_ids)
        cube = self.synapse_cubes[synapse_id]
        
        # Apply different augmentations to create positive pair
        view1 = self._augment_cube(cube)
        view2 = self._augment_cube(cube)
        
        return view1, view2
    
    def _augment_cube(self, cube: torch.Tensor) -> torch.Tensor:
        """Enhanced augmentation function matching SynapseDataset."""
        # Clone the cube to avoid modifying the original
        augmented = cube.clone()
        
        # Random flips along each axis (higher probability)
        if random.random() > 0.3:  # Increased from 0.5
            augmented = torch.flip(augmented, dims=[1])  # flip z
        if random.random() > 0.3:
            augmented = torch.flip(augmented, dims=[2])  # flip y
        if random.random() > 0.3:
            augmented = torch.flip(augmented, dims=[3])  # flip x
        
        # STRONGER ROTATIONS: Not just 90-degree steps
        # Random rotation in xy plane (around z axis) - any angle
        if random.random() > 0.4:
            # For continuous rotation, we'll use discrete steps for simplicity
            k = random.randint(1, 7)  # More rotation options
            augmented = torch.rot90(augmented, k=k, dims=[2, 3])
        
        # Random rotation in xz plane (around y axis)
        if random.random() > 0.4:
            k = random.randint(1, 7)
            augmented = torch.rot90(augmented, k=k, dims=[1, 3])
        
        # Random rotation in yz plane (around x axis)
        if random.random() > 0.4:
            k = random.randint(1, 7)
            augmented = torch.rot90(augmented, k=k, dims=[1, 2])
        
        # ELASTIC DEFORMATION (simplified version using random scaling)
        if random.random() > 0.3:
            augmented = self._apply_elastic_deformation(augmented)
        
        # RANDOM CROPPING AND SCALING
        if random.random() > 0.4:
            augmented = self._random_crop_and_scale(augmented)
        
        # Apply stronger noise and intensity variations to the raw channel (channel 0)
        raw_channel = augmented[0]  # Raw EM intensity
        
        # STRONGER Gaussian noise
        if random.random() > 0.3:  # More frequent
            noise_std = random.uniform(0.02, 0.15)  # Increased from 0.01-0.05
            noise = torch.randn_like(raw_channel) * noise_std
            augmented[0] = torch.clamp(raw_channel + noise, 0, 1)
        
        # STRONGER intensity variations (contrast and brightness)
        if random.random() > 0.3:
            # Stronger contrast: wider range
            contrast_factor = random.uniform(0.5, 1.8)  # Increased from 0.8-1.2
            # Stronger brightness: wider range
            brightness_offset = random.uniform(-0.3, 0.3)  # Increased from -0.1, 0.1
            
            adjusted = raw_channel * contrast_factor + brightness_offset
            augmented[0] = torch.clamp(adjusted, 0, 1)
        
        # GAMMA CORRECTION for additional intensity variation
        if random.random() > 0.4:
            gamma = random.uniform(0.5, 2.0)
            augmented[0] = torch.pow(augmented[0], gamma)
        
        # SALT AND PEPPER NOISE
        if random.random() > 0.6:
            augmented = self._add_salt_pepper_noise(augmented)
        
        return augmented
    
    def _apply_elastic_deformation(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Apply elastic deformation by random local scaling.
        Simplified version - randomly scale different regions.
        """
        C, D, H, W = cube.shape
        
        # Create random scaling factors for different regions
        if random.random() > 0.5:
            # Random scaling in depth
            scale_factor = random.uniform(0.8, 1.2)
            new_depth = max(int(D * scale_factor), D // 2)
            
            # Interpolate along depth dimension
            indices = torch.linspace(0, D-1, new_depth)
            indices = indices.long().clamp(0, D-1)
            
            scaled = cube[:, indices, :, :]
            
            # Resize back to original depth if needed
            if new_depth != D:
                # Simple repetition/subsampling to get back to original size
                if new_depth < D:
                    # Repeat frames
                    repeat_factor = D // new_depth + 1
                    scaled = scaled.repeat(1, repeat_factor, 1, 1)[:, :D, :, :]
                else:
                    # Subsample
                    scaled = scaled[:, :D, :, :]
            
            cube = scaled
        
        return cube
    
    def _random_crop_and_scale(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Randomly crop a region and scale it back to original size.
        """
        C, D, H, W = cube.shape
        
        # Random crop size (between 70% and 95% of original)
        crop_ratio = random.uniform(0.7, 0.95)
        
        crop_d = max(int(D * crop_ratio), D // 2)
        crop_h = max(int(H * crop_ratio), H // 2)
        crop_w = max(int(W * crop_ratio), W // 2)
        
        # Random crop position
        start_d = random.randint(0, D - crop_d)
        start_h = random.randint(0, H - crop_h)
        start_w = random.randint(0, W - crop_w)
        
        # Crop
        cropped = cube[:, start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Scale back using nearest neighbor interpolation (simple repetition)
        # For EM data, we want to preserve discrete values
        scaled = torch.nn.functional.interpolate(
            cropped.unsqueeze(0), 
            size=(D, H, W), 
            mode='nearest'
        ).squeeze(0)
        
        return scaled
    
    def _add_salt_pepper_noise(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Add salt and pepper noise to the raw channel only.
        """
        raw_channel = cube[0].clone()
        
        # Salt and pepper noise
        noise_ratio = random.uniform(0.001, 0.01)  # Small amount
        
        # Salt noise (set random pixels to 1)
        salt_mask = torch.rand_like(raw_channel) < noise_ratio / 2
        raw_channel[salt_mask] = 1.0
        
        # Pepper noise (set random pixels to 0)
        pepper_mask = torch.rand_like(raw_channel) < noise_ratio / 2
        raw_channel[pepper_mask] = 0.0
        
        cube[0] = raw_channel
        return cube 