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
        Apply augmentations to a synapse cube.
        
        Augmentations include:
        - Random flips along each axis
        - Random 90-degree rotations in xy, xz, yz planes
        - Gaussian noise (only to raw channel)
        - Intensity variations (only to raw channel)
        """
        # Clone the cube to avoid modifying the original
        augmented = cube.clone()
        
        # Random flips along each axis
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[1])  # flip z
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[2])  # flip y
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[3])  # flip x
        
        # Random 90-degree rotations
        # Rotate in xy plane (around z axis)
        if random.random() > 0.5:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            augmented = torch.rot90(augmented, k=k, dims=[2, 3])
        
        # Rotate in xz plane (around y axis)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            augmented = torch.rot90(augmented, k=k, dims=[1, 3])
        
        # Rotate in yz plane (around x axis)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            augmented = torch.rot90(augmented, k=k, dims=[1, 2])
        
        # Apply noise and intensity variations only to the raw channel (channel 0)
        raw_channel = augmented[0]  # Raw EM intensity
        
        # Add Gaussian noise
        if random.random() > 0.5:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(raw_channel) * noise_std
            augmented[0] = torch.clamp(raw_channel + noise, 0, 1)
        
        # Intensity variations (contrast and brightness)
        if random.random() > 0.5:
            # Contrast: multiply by a factor
            contrast_factor = random.uniform(0.8, 1.2)
            # Brightness: add an offset
            brightness_offset = random.uniform(-0.1, 0.1)
            
            adjusted = raw_channel * contrast_factor + brightness_offset
            augmented[0] = torch.clamp(adjusted, 0, 1)
        
        return augmented
    
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
        """Same augmentation function as SynapseDataset."""
        # Clone the cube to avoid modifying the original
        augmented = cube.clone()
        
        # Random flips along each axis
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[1])  # flip z
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[2])  # flip y
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[3])  # flip x
        
        # Random 90-degree rotations
        # Rotate in xy plane (around z axis)
        if random.random() > 0.5:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            augmented = torch.rot90(augmented, k=k, dims=[2, 3])
        
        # Rotate in xz plane (around y axis)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            augmented = torch.rot90(augmented, k=k, dims=[1, 3])
        
        # Rotate in yz plane (around x axis)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            augmented = torch.rot90(augmented, k=k, dims=[1, 2])
        
        # Apply noise and intensity variations only to the raw channel (channel 0)
        raw_channel = augmented[0]  # Raw EM intensity
        
        # Add Gaussian noise
        if random.random() > 0.5:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(raw_channel) * noise_std
            augmented[0] = torch.clamp(raw_channel + noise, 0, 1)
        
        # Intensity variations (contrast and brightness)
        if random.random() > 0.5:
            # Contrast: multiply by a factor
            contrast_factor = random.uniform(0.8, 1.2)
            # Brightness: add an offset
            brightness_offset = random.uniform(-0.1, 0.1)
            
            adjusted = raw_channel * contrast_factor + brightness_offset
            augmented[0] = torch.clamp(adjusted, 0, 1)
        
        return augmented 