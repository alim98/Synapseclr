#!/usr/bin/env python3
"""
Preprocess synapse cubes and save them to H5 files for faster loading.

This script extracts all synapses from Excel coordinate data and saves them
as H5 files for much faster loading during training.
"""

import argparse
import os
from datasets.synapse_loader import SynapseLoader

def main():
    parser = argparse.ArgumentParser(description='Preprocess synapse cubes and save to H5 files')
    parser.add_argument('--data_dir', type=str, default='datareal',
                        help='Path to data directory containing raw, seg folders and Excel files')
    parser.add_argument('--preproc_dir', type=str, default='preproc_synapses',
                        help='Directory to store preprocessed synapse H5 files')
    parser.add_argument('--cube_size', type=int, default=80,
                        help='Size of synapse cubes to extract')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing even if H5 files exist')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SYNAPSE PREPROCESSING")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Preprocessing directory: {args.preproc_dir}")
    print(f"Cube size: {args.cube_size}")
    print(f"Force reprocessing: {args.force_reprocess}")
    print("-"*60)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' does not exist!")
        return 1
    
    # Initialize the synapse loader
    loader = SynapseLoader(
        data_dir=args.data_dir,
        preproc_dir=args.preproc_dir,
        cube_size=args.cube_size,
        create_h5=True
    )
    
    # Remove existing H5 files if force reprocessing
    if args.force_reprocess:
        print("Force reprocessing: removing existing H5 files...")
        for bbox_name in loader.bbox_names:
            h5_file = os.path.join(args.preproc_dir, f"{bbox_name}_synapses.h5")
            if os.path.exists(h5_file):
                os.remove(h5_file)
                print(f"  Removed {h5_file}")
    
    # Process all synapses
    print("\nProcessing synapses...")
    synapses = loader.process_all_synapses(memory_efficient=False)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    
    if synapses:
        print(f"Successfully processed {len(synapses)} synapse cubes")
        
        # Print distribution by bbox
        bbox_counts = {}
        total_size_mb = 0
        
        for synapse_id in synapses.keys():
            bbox_name = synapse_id.split('_')[0] + synapse_id.split('_')[1]  # e.g., bbox1
            bbox_counts[bbox_name] = bbox_counts.get(bbox_name, 0) + 1
        
        print("\nSynapse distribution:")
        for bbox_name, count in sorted(bbox_counts.items()):
            h5_file = os.path.join(args.preproc_dir, f"{bbox_name}_synapses.h5")
            if os.path.exists(h5_file):
                size_mb = os.path.getsize(h5_file) / (1024 * 1024)
                total_size_mb += size_mb
                print(f"  {bbox_name}: {count} synapses ({size_mb:.1f} MB)")
            else:
                print(f"  {bbox_name}: {count} synapses (H5 file not found)")
        
        print(f"\nTotal H5 files size: {total_size_mb:.1f} MB")
        print(f"Files saved in: {args.preproc_dir}/")
        
        print("\nYou can now train much faster using:")
        print(f"  python train_synapse_simclr.py --data_dir {args.data_dir} --preproc_dir {args.preproc_dir}")
        
    else:
        print("No synapses were processed successfully!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 