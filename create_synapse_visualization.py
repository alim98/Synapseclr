#!/usr/bin/env python3
"""
Script to create UMAP/t-SNE visualization with GIFs directly from trained checkpoint.

This script extracts embeddings from your trained SimCLR model and creates
interactive visualizations with synapse GIFs.

Usage:
    python create_synapse_visualization.py --checkpoint checkpoints/checkpoint_best.pt
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Create synapse visualization from trained checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained SimCLR checkpoint')
    parser.add_argument('--data_dir', type=str, default='datareal',
                        help='Directory containing raw data')
    parser.add_argument('--preproc_dir', type=str, default='preproc_synapses',
                        help='Directory containing preprocessed H5 files')
    parser.add_argument('--output_dir', type=str, default='synapse_visualization',
                        help='Output directory for visualization')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to process for embeddings')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to create GIFs for')
    parser.add_argument('--dim_reduction', type=str, default='umap', choices=['umap', 'tsne'],
                        help='Dimensionality reduction method')
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters for K-means')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of the model')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Output dimension of the projection head')
    parser.add_argument('--cube_size', type=int, default=80,
                        help='Size of synapse cubes')
    
    args = parser.parse_args()
    
    # Check if checkpoint file exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found!")
        print("Make sure your model has been trained and checkpoint saved.")
        return 1
    
    # Check if H5 files exist
    preproc_path = Path(args.preproc_dir)
    if not preproc_path.exists():
        print(f"Error: Preprocessing directory {args.preproc_dir} not found!")
        return 1
    
    h5_files = list(preproc_path.glob("bbox*_synapses.h5"))
    if not h5_files:
        print(f"Error: No H5 files found in {args.preproc_dir}!")
        print("Make sure your synapse data is preprocessed.")
        return 1
    
    print(f"Found {len(h5_files)} H5 files in {args.preproc_dir}")
    print(f"Creating visualization from checkpoint: {args.checkpoint}")
    
    # Run the visualization
    script_path = Path(__file__).parent / "umaputil" / "GifUmap.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--checkpoint", args.checkpoint,
        "--data_dir", args.data_dir,
        "--preproc_dir", args.preproc_dir,
        "--output_dir", args.output_dir,
        "--max_samples", str(args.max_samples),
        "--num_samples", str(args.num_samples),
        "--dim_reduction", args.dim_reduction,
        "--n_clusters", str(args.n_clusters),
        "--hidden_dim", str(args.hidden_dim),
        "--output_dim", str(args.output_dim),
        "--cube_size", str(args.cube_size)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\nVisualization completed successfully!")
        print(f"Output saved to: {args.output_dir}")
        print(f"Open the HTML file in your browser to view the interactive visualization.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running visualization: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 