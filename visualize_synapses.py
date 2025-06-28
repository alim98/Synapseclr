#!/usr/bin/env python3
"""
Visualization script for synapse data that saves PNG images.
Shows EM images, semantic segmentation, and instance segmentation.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
import argparse
from pathlib import Path

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def visualize_synapse_slices(synapse_data, synapse_id, output_dir, num_slices=5):
    """
    Visualize multiple slices of a synapse showing EM, semantic seg, and instance seg.
    
    Args:
        synapse_data: 4D numpy array (3, depth, height, width)
        synapse_id: string identifier for this synapse
        output_dir: directory to save images
        num_slices: number of evenly spaced slices to visualize
    """
    depth = synapse_data.shape[1]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    # Create figure with subplots for each slice
    fig, axes = plt.subplots(num_slices, 3, figsize=(12, 4*num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    # Define custom colormap for segmentation
    seg_colors = ['black', 'red']  # 0=background, 1=synapse
    seg_cmap = ListedColormap(seg_colors)
    
    for i, slice_idx in enumerate(slice_indices):
        # EM image (channel 0)
        em_slice = synapse_data[0, slice_idx, :, :]
        axes[i, 0].imshow(em_slice, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'EM Image - Slice {slice_idx}')
        axes[i, 0].axis('off')
        
        # Semantic segmentation (channel 1)
        sem_slice = synapse_data[1, slice_idx, :, :]
        axes[i, 1].imshow(sem_slice, cmap=seg_cmap, vmin=0, vmax=1)
        axes[i, 1].set_title(f'Semantic Seg - Slice {slice_idx}')
        axes[i, 1].axis('off')
        
        # Instance segmentation (channel 2)
        inst_slice = synapse_data[2, slice_idx, :, :]
        axes[i, 2].imshow(inst_slice, cmap=seg_cmap, vmin=0, vmax=1)
        axes[i, 2].set_title(f'Instance Seg - Slice {slice_idx}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'{synapse_id}_slices.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {output_path}")

def visualize_synapse_overview(synapse_data, synapse_id, output_dir):
    """
    Create an overview visualization showing middle slice and 3D projections.
    
    Args:
        synapse_data: 4D numpy array (3, depth, height, width)
        synapse_id: string identifier for this synapse
        output_dir: directory to save images
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Middle slice for each channel
    middle_depth = synapse_data.shape[1] // 2
    
    # Top row: Middle slices
    em_slice = synapse_data[0, middle_depth, :, :]
    sem_slice = synapse_data[1, middle_depth, :, :]
    inst_slice = synapse_data[2, middle_depth, :, :]
    
    axes[0, 0].imshow(em_slice, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'EM - Middle Slice ({middle_depth})')
    axes[0, 0].axis('off')
    
    seg_cmap = ListedColormap(['black', 'red'])
    axes[0, 1].imshow(sem_slice, cmap=seg_cmap, vmin=0, vmax=1)
    axes[0, 1].set_title(f'Semantic Seg - Middle Slice')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(inst_slice, cmap=seg_cmap, vmin=0, vmax=1)
    axes[0, 2].set_title(f'Instance Seg - Middle Slice')
    axes[0, 2].axis('off')
    
    # Bottom row: Max projections (Z-axis)
    em_proj = np.max(synapse_data[0], axis=0)
    sem_proj = np.max(synapse_data[1], axis=0)
    inst_proj = np.max(synapse_data[2], axis=0)
    
    axes[1, 0].imshow(em_proj, cmap='gray')
    axes[1, 0].set_title('EM - Max Z Projection')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sem_proj, cmap=seg_cmap, vmin=0, vmax=1)
    axes[1, 1].set_title('Semantic Seg - Max Z Projection')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(inst_proj, cmap=seg_cmap, vmin=0, vmax=1)
    axes[1, 2].set_title('Instance Seg - Max Z Projection')
    axes[1, 2].axis('off')
    
    # Add overall title
    fig.suptitle(f'Synapse: {synapse_id}\nShape: {synapse_data.shape}', fontsize=14)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'{synapse_id}_overview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved overview: {output_path}")

def visualize_synapse_file(filepath, output_dir, max_synapses=5, num_slices=5):
    """
    Visualize synapses from an H5 file and save as PNG images.
    
    Args:
        filepath: path to H5 file
        output_dir: directory to save images
        max_synapses: maximum number of synapses to visualize
        num_slices: number of slices to show per synapse
    """
    print(f"=== VISUALIZING: {filepath} ===")
    
    # Create output directory
    create_output_dir(output_dir)
    
    with h5py.File(filepath, 'r') as f:
        synapse_ids = list(f.keys())[:max_synapses]
        print(f"Total synapses in file: {len(f.keys())}")
        print(f"Visualizing first {len(synapse_ids)} synapses")
        
        for synapse_id in synapse_ids:
            print(f"\nProcessing: {synapse_id}")
            synapse_data = f[synapse_id][...]
            
            # Create both overview and slice visualizations
            visualize_synapse_overview(synapse_data, synapse_id, output_dir)
            visualize_synapse_slices(synapse_data, synapse_id, output_dir, num_slices)
    
    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"Images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize synapse data and save as PNG images')
    parser.add_argument('--input_file', type=str, default='preproc_synapses/bbox1_synapses.h5',
                        help='Path to H5 file containing synapse data')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualization images')
    parser.add_argument('--max_synapses', type=int, default=5,
                        help='Maximum number of synapses to visualize')
    parser.add_argument('--num_slices', type=int, default=5,
                        help='Number of slices to show per synapse')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found!")
        return
    
    visualize_synapse_file(args.input_file, args.output_dir, args.max_synapses, args.num_slices)

if __name__ == "__main__":
    main() 