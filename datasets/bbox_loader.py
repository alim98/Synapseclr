import os
import glob
import numpy as np
import torch
import h5py
import pandas as pd
from typing import Tuple, Dict, Optional, List
from scipy import ndimage
import imageio.v2 as iio

class BBoxLoader:
    """
    Loads and preprocesses the bbox data from raw, seg, and additional mask directories.
    
    Handles:
    1. Loading TIFF slices from different data sources
    2. Standardizing label IDs across bboxes
    3. Isolating the vesicle cloud closest to the cleft
    4. Determining pre vs post synaptic sides
    5. Converting to 5-channel tensor format
    6. Normalizing the raw channel
    7. Optional saving to H5 for faster subsequent loading
    """
    
    # Map from bbox number to the label values used in that bbox's annotation files
    LABEL_MAP = {
        '1': {'mito': 5, 'vesicle': 6, 'cleft': 7, 'cleft2': 7},
        '2': {'mito': 1, 'vesicle': 3, 'cleft': 2, 'cleft2': 4},
        '3': {'mito': 6, 'vesicle': 7, 'cleft': 9, 'cleft2': 8},
        '4': {'mito': 3, 'vesicle': 2, 'cleft': 1, 'cleft2': 4},
        '5': {'mito': 1, 'vesicle': 3, 'cleft': 2, 'cleft2': 4},
        '6': {'mito': 5, 'vesicle': 6, 'cleft': 7, 'cleft2': 7},
        '7': {'mito': 1, 'vesicle': 2, 'cleft': 4, 'cleft2': 3},
    }
    
    def __init__(self, data_dir: str = 'data', 
                 preproc_dir: str = 'preproc',
                 create_h5: bool = True):
        """
        Initialize the BBoxLoader.
        
        Args:
            data_dir: Root directory for data
            preproc_dir: Directory to store preprocessed H5 files
            create_h5: Whether to save preprocessed volumes to H5 files
        """
        self.data_dir = data_dir
        self.preproc_dir = preproc_dir
        self.create_h5 = create_h5
        
        # Directory paths
        self.raw_base_dir = os.path.join(data_dir, 'raw')
        self.seg_base_dir = os.path.join(data_dir, 'seg')
        self.add_mask_base_dir = data_dir  # bbox_X directories are in the root data dir
        self.bbox_names = [f'bbox{i}' for i in range(1, 8)]
        
        if create_h5:
            os.makedirs(preproc_dir, exist_ok=True)
    
    def load_excel_data(self, bbox_name: str) -> pd.DataFrame:
        """Load Excel data for the given bbox."""
        bbox_num = bbox_name.replace("bbox", "")
        excel_file = os.path.join(self.data_dir, f"{bbox_name}.xlsx")
        try:
            df = pd.read_excel(excel_file)
            return df
        except Exception as e:
            print(f"Error loading Excel file for {bbox_name}: {e}")
            return None
    
    def load_volumes(self, bbox_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw, segmentation, and additional mask volumes for a given bbox.
        
        Returns:
            Tuple of (raw_volume, segmentation_volume, additional_mask_volume)
        """
        # Define paths
        raw_dir = os.path.join(self.raw_base_dir, bbox_name)
        seg_dir = os.path.join(self.seg_base_dir, bbox_name)
        
        bbox_num = bbox_name.replace("bbox", "")
        add_mask_dir = os.path.join(self.add_mask_base_dir, f"bbox_{bbox_num}")
        
        # Find TIFF files
        raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
        seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
        add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))
        
        # Check that we have the same number of files for each volume
        if not (len(raw_tif_files) == len(seg_tif_files) == len(add_mask_tif_files)):
            print(f"Warning: Different number of files in the volumes for {bbox_name}")
            print(f"Raw: {len(raw_tif_files)}, Seg: {len(seg_tif_files)}, Add: {len(add_mask_tif_files)}")
            return None, None, None
        
        try:
            # Load raw volume and convert to grayscale if needed
            raw_slices = []
            multi_channel_detected = False
            for f in raw_tif_files:
                img = iio.imread(f)
                # Check if the image has multiple channels (RGB)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # Convert RGB to grayscale using luminosity method
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                raw_slices.append(img)
            raw_vol = np.stack(raw_slices, axis=0)
            
            # Load segmentation volume and ensure it's single channel
            seg_slices = []
            for f in seg_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # For segmentation, take first channel (labels should be consistent)
                    img = img[..., 0]
                seg_slices.append(img.astype(np.uint32))
            seg_vol = np.stack(seg_slices, axis=0)
            
            # Load additional mask volume and ensure it's single channel
            add_mask_slices = []
            for f in add_mask_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # For masks, take first channel
                    img = img[..., 0]
                add_mask_slices.append(img.astype(np.uint32))
            add_mask_vol = np.stack(add_mask_slices, axis=0)
            
            if multi_channel_detected:
                print(f"WARNING: Multi-channel images detected in {bbox_name} and converted to single-channel")
            
            return raw_vol, seg_vol, add_mask_vol
        except Exception as e:
            print(f"Error loading volumes for {bbox_name}: {e}")
            return None, None, None
    
    def get_closest_component_mask(self, binary_mask: np.ndarray, 
                                   cleft_center: Tuple[int, int, int]) -> np.ndarray:
        """
        Find the connected component in the binary mask that is closest to the cleft center.
        
        Args:
            binary_mask: Binary mask of all vesicle clouds
            cleft_center: (x, y, z) coordinates of the cleft center
            
        Returns:
            Binary mask containing only the closest vesicle cloud component
        """
        # Transpose cleft center to match array dimensions (z, y, x)
        cx, cy, cz = cleft_center
        cleft_center_zyx = (cz, cy, cx)
        
        # Find connected components in the binary mask
        labeled_mask, num_features = ndimage.label(binary_mask)
        
        if num_features == 0:
            print("Warning: No components found in vesicle mask")
            return np.zeros_like(binary_mask)
        
        if num_features == 1:
            return binary_mask
        
        # Find the centroid of each component
        component_centroids = ndimage.center_of_mass(binary_mask, labeled_mask, 
                                                    range(1, num_features + 1))
        
        # Calculate Euclidean distance from each centroid to the cleft center
        distances = [np.linalg.norm(np.array(centroid) - np.array(cleft_center_zyx)) 
                    for centroid in component_centroids]
        
        # Check for NaN distances (when centroid is outside the crop)
        valid_indices = [i for i, d in enumerate(distances) if not np.isnan(d)]
        
        # If all distances are NaN (all centroids outside crop), keep all vesicle voxels
        if len(valid_indices) == 0:
            print("Warning: All vesicle component centroids are outside crop - keeping all vesicle voxels")
            return binary_mask
        
        # Get valid distances only
        valid_distances = [distances[i] for i in valid_indices]
        
        # Find the valid component with minimum distance
        min_distance_idx = np.argmin(valid_distances)
        closest_component_idx = valid_indices[min_distance_idx] + 1  # +1 because labels start at 1
        
        # Create a mask containing only the closest component
        closest_component_mask = (labeled_mask == closest_component_idx)
        
        return closest_component_mask
    
    def create_segment_masks(self, segmentation_volume: np.ndarray, 
                            s1_coord: Tuple[int, int, int], 
                            s2_coord: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binary masks for the two sides from the segmentation volume.
        
        Args:
            segmentation_volume: Volume with segmentation labels
            s1_coord: (x, y, z) coordinate for side 1
            s2_coord: (x, y, z) coordinate for side 2
            
        Returns:
            Tuple of binary masks for side 1 and side 2
        """
        x1, y1, z1 = s1_coord
        x2, y2, z2 = s2_coord
        
        # Get the segment IDs for both sides
        try:
            seg_id_1 = segmentation_volume[z1, y1, x1]
            seg_id_2 = segmentation_volume[z2, y2, x2]
            
            # Create binary masks for each side
            mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            
            return mask_1, mask_2
        except IndexError as e:
            print(f"Error creating segment masks: {e}")
            return np.zeros_like(segmentation_volume, dtype=bool), np.zeros_like(segmentation_volume, dtype=bool)
    
    def process_bbox(self, bbox_name: str) -> Optional[torch.Tensor]:
        """
        Process a single bbox into a 3-channel tensor.
        
        Returns:
            torch.Tensor of shape (3, D, H, W) containing:
            [0] - Raw EM intensity (normalized)
            [1] - Pre-synaptic mask
            [2] - Cleft mask
        """
        # Check if H5 file already exists
        h5_file = os.path.join(self.preproc_dir, f"{bbox_name}.h5") 
        if self.create_h5 and os.path.exists(h5_file):
            try:
                with h5py.File(h5_file, 'r') as f:
                    volume = torch.from_numpy(f['volume'][()]).float()  # Explicitly convert to float32
                    print(f"Loaded preprocessed {bbox_name} from H5 file")
                    return volume
            except Exception as e:
                print(f"Error loading H5 file for {bbox_name}: {e}")
        
        # Load the volumes
        raw_vol, seg_vol, add_mask_vol = self.load_volumes(bbox_name)
        if raw_vol is None or seg_vol is None or add_mask_vol is None:
            return None
        
        # Load Excel data to get cleft and side coordinates
        excel_data = self.load_excel_data(bbox_name)
        if excel_data is None:
            return None
            
        # Extract coordinates
        # Excel columns: Var1, central_coord_1, central_coord_2, central_coord_3, 
        #                side_1_coord_1, side_1_coord_2, side_1_coord_3, 
        #                side_2_coord_1, side_2_coord_2, side_2_coord_3
        # First row should have these coordinates
        try:
            row = excel_data.iloc[0]
            cleft_x = int(row['central_coord_1'])
            cleft_y = int(row['central_coord_2'])
            cleft_z = int(row['central_coord_3'])
            
            side1_x = int(row['side_1_coord_1'])
            side1_y = int(row['side_1_coord_2'])
            side1_z = int(row['side_1_coord_3'])
            
            side2_x = int(row['side_2_coord_1']) 
            side2_y = int(row['side_2_coord_2'])
            side2_z = int(row['side_2_coord_3'])
        except (KeyError, ValueError) as e:
            print(f"Error extracting coordinates from Excel for {bbox_name}: {e}")
            print("Excel columns should include: central_coord_[1-3], side_[1-2]_coord_[1-3]")
            return None
            
        # Get the label values for this bbox
        bbox_num = bbox_name.replace("bbox", "")
        label_map = self.LABEL_MAP.get(bbox_num, 
                                      {'mito': 5, 'vesicle': 6, 'cleft': 7, 'cleft2': 7})
        
        cleft_label = label_map['cleft']
        cleft_label2 = label_map['cleft2']
        
        # Create binary masks for each structure
        cleft_mask = ((add_mask_vol == cleft_label) | (add_mask_vol == cleft_label2))
        
        # Get segmentation masks for the two sides
        side1_coord = (side1_x, side1_y, side1_z)
        side2_coord = (side2_x, side2_y, side2_z)
        mask_1_full, mask_2_full = self.create_segment_masks(seg_vol, side1_coord, side2_coord)
        
        # Get pre-synapse mask - we'll use first side as pre-synaptic side
        presynapse_mask = mask_1_full
        
        # Normalize the raw volume (z-score per cube)
        raw_vol_norm = (raw_vol - np.mean(raw_vol)) / (np.std(raw_vol) + 1e-6)
        
        # Create the 3-channel volume
        # [0] - Raw EM intensity (normalized)
        # [1] - Pre-synaptic mask
        # [2] - Cleft mask
        volume = np.stack([
            raw_vol_norm,
            presynapse_mask.astype(np.float32),
            cleft_mask.astype(np.float32)
        ], axis=0)
        
        # Convert to PyTorch tensor
        volume_tensor = torch.from_numpy(volume).float()  # Explicitly convert to float32
        
        # Save to H5 file for faster loading next time
        if self.create_h5:
            try:
                with h5py.File(h5_file, 'w') as f:
                    f.create_dataset('volume', data=volume.astype(np.float32))  # Save as float32
                print(f"Saved preprocessed {bbox_name} to H5 file")
            except Exception as e:
                print(f"Error saving H5 file for {bbox_name}: {e}")
        
        return volume_tensor
    
    def process_all_bboxes(self, memory_efficient: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process all bboxes and return a dictionary of tensors.
        
        Args:
            memory_efficient: If True, process one bbox at a time and clear memory between each.
                              This reduces peak memory usage but might be slower.
        
        Returns:
            Dict mapping bbox_name -> 3-channel tensor
        """
        result = {}
        for bbox_name in self.bbox_names:
            print(f"Processing {bbox_name}...")
            try:
                # Process the current bbox
                tensor = self.process_bbox(bbox_name)
                if tensor is not None:
                    # If memory_efficient is True, store only the path to H5 file instead of keeping tensor in memory
                    if memory_efficient and self.create_h5:
                        # Store reference to h5 file path
                        h5_path = os.path.join(self.preproc_dir, f"{bbox_name}.h5")
                        if os.path.exists(h5_path):
                            result[bbox_name] = h5_path
                            # Free memory by deleting the tensor
                            del tensor
                            # Force garbage collection
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print(f"Stored reference to {bbox_name} H5 file to save memory")
                        else:
                            print(f"Warning: H5 file for {bbox_name} not created. Keeping tensor in memory.")
                            result[bbox_name] = tensor
                    else:
                        # Store the tensor directly
                        result[bbox_name] = tensor
            except Exception as e:
                print(f"Error processing {bbox_name}: {e}")
                # Continue processing other bboxes even if one fails
                continue
        
        return result

    def load_from_paths(self, paths_dict: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Load tensors from H5 file paths for memory-efficient loading.
        
        Args:
            paths_dict: Dictionary mapping bbox_name -> h5_file_path
            
        Returns:
            Dictionary mapping bbox_name -> tensor
        """
        result = {}
        for bbox_name, h5_path in paths_dict.items():
            try:
                with h5py.File(h5_path, 'r') as f:
                    tensor = torch.from_numpy(f['volume'][()]).float()
                    result[bbox_name] = tensor
                    print(f"Loaded {bbox_name} from H5 file")
            except Exception as e:
                print(f"Error loading {bbox_name} from H5 file: {e}")
        
        return result


def load_tiff_stack(path_pattern: str) -> np.ndarray:
    """
    Load a stack of TIFF files into a single 3D array.
    
    Args:
        path_pattern: Glob pattern for the TIFF files
        
    Returns:
        3D numpy array with the stack data
    """
    tif_files = sorted(glob.glob(path_pattern))
    
    if not tif_files:
        raise ValueError(f"No TIFF files found matching pattern: {path_pattern}")
    
    slices = []
    for f in tif_files:
        img = iio.imread(f)
        # Convert to grayscale if needed
        if len(img.shape) > 2 and img.shape[2] > 1:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        slices.append(img)
    
    return np.stack(slices, axis=0)


if __name__ == "__main__":
    # Example usage
    loader = BBoxLoader(data_dir='data', preproc_dir='preproc', create_h5=True)
    volumes = loader.process_all_bboxes()
    
    # Print shapes of all volumes
    for bbox_name, tensor in volumes.items():
        print(f"{bbox_name} shape: {tensor.shape}")
