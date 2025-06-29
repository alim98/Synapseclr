#!/usr/bin/env python3
"""
Minimal test to force HTML creation
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append('.')

# Import the function directly
from umaputil.GifUmap import create_animated_gif_visualization

def main():
    print("=== Testing HTML Creation ===")
    
    # Create minimal test data
    features_df = pd.DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [1.0, 2.0, 3.0], 
        'cluster': [0, 1, 0],
        'bbox_name': ['bbox1', 'bbox2', 'bbox1'],
        'synapse_name': ['test1', 'test2', 'test3'],
        'Var1': ['test1', 'test2', 'test3']
    })
    
    # Check if we have real GIF files
    gif_dir = Path('debug_viz/sample_gifs')
    if gif_dir.exists():
        gif_files = list(gif_dir.glob('*.gif'))
        print(f"Found {len(gif_files)} GIF files")
        
        if len(gif_files) >= 3:
            # Create gif_paths using first 3 files
            gif_paths = {i: str(gif_files[i].absolute()) for i in range(3)}
            
            # Create minimal frame data
            frame_data = {i: [f"frame_{i}_{j}" for j in range(5)] for i in range(3)}
            
            # Create minimal max_slices_data
            max_slices_data = {i: {'xy': 40, 'xz': 40, 'yz': 40} for i in range(3)}
            
            output_dir = Path('force_html_test_output')
            output_dir.mkdir(exist_ok=True)
            
            print(f"Calling create_animated_gif_visualization...")
            print(f"  features_df shape: {features_df.shape}")
            print(f"  gif_paths keys: {list(gif_paths.keys())}")
            print(f"  output_dir: {output_dir}")
            
            try:
                result = create_animated_gif_visualization(
                    features_df=features_df,
                    gif_paths=gif_paths,
                    output_dir=output_dir,
                    dim_reduction='umap',
                    frame_data=frame_data,
                    max_slices_data=max_slices_data
                )
                
                print(f"SUCCESS! Result: {result}")
                
                # Check what files were created
                if output_dir.exists():
                    files = list(output_dir.glob('*'))
                    print(f"Files created: {[f.name for f in files]}")
                    
                    for f in files:
                        if f.suffix == '.html':
                            print(f"HTML file size: {f.stat().st_size} bytes")
                            
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Not enough GIF files to test with")
    else:
        print("No GIF directory found")

if __name__ == "__main__":
    main() 