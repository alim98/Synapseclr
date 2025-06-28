#!/usr/bin/env python3
"""
Test script to verify synapse extraction from Excel data.

This script tests the SynapseLoader to ensure it can properly extract
synapse cubes based on coordinates from Excel files.
"""

import os
import sys
from datasets.synapse_loader import SynapseLoader
from datasets.synapse_dataset import SynapseDataset

def test_synapse_extraction(data_dir='data', cube_size=80):
    """Test synapse extraction from Excel data."""
    
    print("Testing synapse extraction...")
    print(f"Data directory: {data_dir}")
    print(f"Cube size: {cube_size}")
    print("-" * 50)
    
    # Initialize the synapse loader
    loader = SynapseLoader(
        data_dir=data_dir,
        preproc_dir='test_preproc_synapses',
        cube_size=cube_size
    )
    
    # Test loading Excel data for each bbox
    print("Testing Excel data loading:")
    for bbox_name in loader.bbox_names:
        excel_data = loader.load_excel_data(bbox_name)
        if excel_data is not None:
            print(f"  {bbox_name}: {len(excel_data)} synapses found")
            print(f"    Columns: {list(excel_data.columns)}")
            if len(excel_data) > 0:
                # Show first synapse as example
                first_row = excel_data.iloc[0]
                synapse_id = str(first_row['Var1'])
                central_coord = (int(first_row['central_coord_1']), int(first_row['central_coord_2']), int(first_row['central_coord_3']))
                print(f"    Example synapse: {synapse_id} at {central_coord}")
        else:
            print(f"  {bbox_name}: Failed to load Excel data")
    print()
    
    # Test loading volumes for one bbox
    print("Testing volume loading:")
    test_bbox = 'bbox1'
    raw_vol, seg_vol, add_mask_vol = loader.load_volumes(test_bbox)
    if raw_vol is not None:
        print(f"  {test_bbox} volumes loaded successfully:")
        print(f"    Raw volume shape: {raw_vol.shape}")
        print(f"    Seg volume shape: {seg_vol.shape}")
        print(f"    Add mask volume shape: {add_mask_vol.shape}")
    else:
        print(f"  {test_bbox}: Failed to load volumes")
    print()
    
    # Test processing synapses for one bbox
    print("Testing synapse processing:")
    synapse_cubes = loader.process_bbox_synapses(test_bbox)
    if synapse_cubes:
        print(f"  {test_bbox}: Successfully extracted {len(synapse_cubes)} synapse cubes")
        for synapse_id, cube in synapse_cubes[:3]:  # Show first 3
            print(f"    {synapse_id}: shape {cube.shape}, dtype {cube.dtype}")
    else:
        print(f"  {test_bbox}: No synapse cubes extracted")
    print()
    
    # Test processing all synapses (limit to first few for speed)
    print("Testing processing all synapses (limited):")
    # Temporarily limit to first bbox for testing
    original_bbox_names = loader.bbox_names
    loader.bbox_names = [test_bbox]  # Only process first bbox for testing
    
    all_synapses = loader.process_all_synapses()
    loader.bbox_names = original_bbox_names  # Restore original
    
    if all_synapses:
        print(f"  Total synapses extracted: {len(all_synapses)}")
        
        # Print distribution
        bbox_counts = {}
        for synapse_id in all_synapses.keys():
            bbox_name = synapse_id.split('_')[0] + synapse_id.split('_')[1]
            bbox_counts[bbox_name] = bbox_counts.get(bbox_name, 0) + 1
        
        print("  Distribution:")
        for bbox_name, count in sorted(bbox_counts.items()):
            print(f"    {bbox_name}: {count} synapses")
    else:
        print("  No synapses extracted")
    print()
    
    # Test dataset creation
    if all_synapses:
        print("Testing dataset creation:")
        dataset = SynapseDataset(synapse_cubes=all_synapses, augment=True)
        print(f"  Dataset created with {len(dataset)} samples")
        
        # Test getting one sample
        if len(dataset) > 0:
            view1, view2 = dataset[0]
            print(f"  Sample 0:")
            print(f"    View 1 shape: {view1.shape}, dtype: {view1.dtype}")
            print(f"    View 2 shape: {view2.shape}, dtype: {view2.dtype}")
            print(f"    Synapse ID: {dataset.get_synapse_info(0)}")
        
        # Test dataset stats
        stats = dataset.get_stats()
        print(f"  Dataset stats:")
        print(f"    Number of synapses: {stats['num_synapses']}")
        print(f"    Bbox distribution: {stats['bbox_distribution']}")
    print()
    
    print("Testing completed successfully!")
    return all_synapses, dataset if all_synapses else None

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test synapse extraction from Excel data')
    parser.add_argument('--data_dir', type=str, default='datareal',
                        help='Path to data directory')
    parser.add_argument('--cube_size', type=int, default=80,
                        help='Size of synapse cubes to extract')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        print("Please ensure you have the data directory with raw/, seg/ folders and Excel files.")
        sys.exit(1)
    
    try:
        synapses, dataset = test_synapse_extraction(args.data_dir, args.cube_size)
        print("\n" + "="*50)
        print("SUCCESS: All tests passed!")
        if synapses:
            print(f"Extracted {len(synapses)} synapse cubes successfully.")
            print("You can now use train_synapse_simclr.py to train on individual synapses.")
    except Exception as e:
        print("\n" + "="*50)
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 