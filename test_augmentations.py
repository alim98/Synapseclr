#!/usr/bin/env python3
"""
Test script to visualize the enhanced augmentations.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets.synapse_dataset import SynapseDataset

def test_augmentations():
    """Test the enhanced augmentations and save visualizations."""
    
    # Load a synapse sample
    with h5py.File("preproc_synapses/bbox1_synapses.h5", 'r') as f:
        synapse_id = list(f.keys())[0]
        synapse_data = torch.tensor(f[synapse_id][...])
    
    print(f"Testing augmentations on: {synapse_id}")
    print(f"Original shape: {synapse_data.shape}")
    
    # Create dataset for augmentations
    synapse_cubes = {synapse_id: synapse_data}
    dataset = SynapseDataset(synapse_cubes, augment=True)
    
    # Generate multiple augmented versions
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    # Original (middle slice)
    middle_slice = synapse_data.shape[1] // 2
    original_em = synapse_data[0, middle_slice, :, :]
    
    axes[0, 0].imshow(original_em, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original EM')
    axes[0, 0].axis('off')
    
    # Show 5 different augmented versions
    for i in range(5):
        view1, view2 = dataset[0]  # Get augmented pair
        
        # Show first view
        aug_em = view1[0, middle_slice, :, :]
        axes[0, i+1].imshow(aug_em, cmap='gray', vmin=0, vmax=1)
        axes[0, i+1].set_title(f'Aug {i+1} - View 1')
        axes[0, i+1].axis('off')
        
        # Show second view  
        aug_em2 = view2[0, middle_slice, :, :]
        axes[1, i+1].imshow(aug_em2, cmap='gray', vmin=0, vmax=1)
        axes[1, i+1].set_title(f'Aug {i+1} - View 2')
        axes[1, i+1].axis('off')
        
        # Show difference
        diff = torch.abs(aug_em - aug_em2)
        axes[2, i+1].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, i+1].set_title(f'Difference {i+1}')
        axes[2, i+1].axis('off')
    
    # Fill first column
    axes[1, 0].imshow(original_em, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Original EM')
    axes[1, 0].axis('off')
    
    axes[2, 0].text(0.5, 0.5, 'Enhanced\nAugmentations\nTest', 
                    ha='center', va='center', transform=axes[2, 0].transAxes,
                    fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/augmentation_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Augmentation test saved to: visualizations/augmentation_test.png")
    
    # Test statistics
    print("\n=== AUGMENTATION STATISTICS ===")
    intensities = []
    contrasts = []
    
    for i in range(20):
        view1, view2 = dataset[0]
        em_data = view1[0].flatten()
        
        intensities.append(em_data.mean().item())
        contrasts.append(em_data.std().item())
    
    print(f"Original - Mean: {synapse_data[0].mean():.3f}, Std: {synapse_data[0].std():.3f}")
    print(f"Augmented - Mean range: {min(intensities):.3f} to {max(intensities):.3f}")
    print(f"Augmented - Std range: {min(contrasts):.3f} to {max(contrasts):.3f}")
    print(f"Intensity variation: {(max(intensities) - min(intensities)):.3f}")
    print(f"Contrast variation: {(max(contrasts) - min(contrasts)):.3f}")

if __name__ == "__main__":
    test_augmentations() 