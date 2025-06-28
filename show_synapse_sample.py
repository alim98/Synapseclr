#!/usr/bin/env python3
"""
Quick script to examine the content of preprocessed synapse H5 files.
"""

import h5py
import numpy as np

def show_synapse_file(filepath):
    """Show the content of a synapse H5 file."""
    print(f"=== EXAMINING: {filepath} ===")
    
    with h5py.File(filepath, 'r') as f:
        print(f"Total synapses: {len(f.keys())}")
        print(f"Synapse IDs: {list(f.keys())[:5]}{'...' if len(f.keys()) > 5 else ''}")
        print()
        
        # Pick the first synapse to examine
        first_key = list(f.keys())[0]
        synapse_data = f[first_key][...]
        
        print(f"Example synapse: {first_key}")
        print(f"Shape: {synapse_data.shape} (channels, depth, height, width)")
        print(f"Data type: {synapse_data.dtype}")
        print(f"Overall value range: [{synapse_data.min():.3f}, {synapse_data.max():.3f}]")
        print()
        
        # Show statistics for each channel
        channel_names = ["EM Image", "Semantic Segmentation", "Instance Segmentation"]
        for i in range(3):
            channel_data = synapse_data[i]
            print(f"Channel {i} ({channel_names[i]}):")
            print(f"  Range: [{channel_data.min():.3f}, {channel_data.max():.3f}]")
            print(f"  Mean: {channel_data.mean():.3f}, Std: {channel_data.std():.3f}")
            print(f"  Unique values: {len(np.unique(channel_data))}")
            print()
        
        # Show a sample slice from the middle
        print("=== SAMPLE SLICE FROM MIDDLE OF CUBE ===")
        middle_depth = synapse_data.shape[1] // 2
        em_slice = synapse_data[0, middle_depth, :, :]  # EM channel, middle depth
        
        print(f"EM slice shape: {em_slice.shape}")
        print("Sample 8x8 region from center:")
        center = em_slice.shape[0] // 2
        sample_region = em_slice[center-4:center+4, center-4:center+4]
        
        # Format for better display
        np.set_printoptions(precision=2, suppress=True, linewidth=100)
        print(sample_region)
        print()
        
        # Show semantic segmentation slice
        seg_slice = synapse_data[1, middle_depth, :, :]
        print("Semantic segmentation - unique values in this slice:")
        unique_vals = np.unique(seg_slice)
        print(f"Values: {unique_vals}")
        print(f"Counts: {[np.sum(seg_slice == val) for val in unique_vals]}")

if __name__ == "__main__":
    show_synapse_file("preproc_synapses/bbox1_synapses.h5") 