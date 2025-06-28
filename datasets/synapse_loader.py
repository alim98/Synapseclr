import os
import glob
import numpy as np
import torch
import h5py
import pandas as pd
from typing import Tuple, Dict, Optional, List
from scipy.ndimage import label
import imageio.v2 as iio

class SynapseLoader:
    """
    Loads and preprocesses individual synapse cubes from Excel annotation data.
    
    This loader extracts specific synapses based on coordinates from Excel files.
    Each synapse is defined by:
    - Central coordinate (cleft center) 
    - Side 1 coordinate (one synaptic partner)
    - Side 2 coordinate (other synaptic partner)
    
    The loader determines which side is presynaptic based on overlap with vesicle clouds.
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
                 preproc_dir: str = 'preproc_synapses',
                 cube_size: int = 80,
                 create_h5: bool = True):
        """
        Initialize the SynapseLoader.
        
        Args:
            data_dir: Root directory for data
            preproc_dir: Directory to store preprocessed H5 files
            cube_size: Size of extracted cubes (cube_size x cube_size x cube_size)
            create_h5: Whether to save preprocessed cubes to H5 files
        """
        self.data_dir = data_dir
        self.preproc_dir = preproc_dir
        self.cube_size = cube_size
        self.create_h5 = create_h5
        
        # Directory paths
        self.raw_base_dir = os.path.join(data_dir, 'raw')
        self.seg_base_dir = os.path.join(data_dir, 'seg')
        self.add_mask_base_dir = data_dir  # bbox_X directories are in the root data dir
        self.bbox_names = [f'bbox{i}' for i in range(1, 8)]
        
        if create_h5:
            os.makedirs(preproc_dir, exist_ok=True)
    
    def load_excel_data(self, bbox_name: str) -> Optional[pd.DataFrame]:
        """Load Excel data for the given bbox."""
        excel_file = os.path.join(self.data_dir, f"{bbox_name}.xlsx")
        try:
            df = pd.read_excel(excel_file)
            return df
        except Exception as e:
            print(f"Error loading Excel file for {bbox_name}: {e}")
            return None
    
    def load_volumes(self, bbox_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load raw, segmentation, and additional mask volumes for a given bbox."""
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
            return None, None, None
        
        try:
            # Load raw volume and convert to grayscale if needed
            raw_slices = []
            for f in raw_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    # Convert RGB to grayscale using luminosity method
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                raw_slices.append(img)
            raw_vol = np.stack(raw_slices, axis=0)
            
            # Load segmentation volume and ensure it's single channel
            seg_slices = []
            for f in seg_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    img = img[..., 0]
                seg_slices.append(img.astype(np.uint32))
            seg_vol = np.stack(seg_slices, axis=0)
            
            # Load additional mask volume and ensure it's single channel
            add_mask_slices = []
            for f in add_mask_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    img = img[..., 0]
                add_mask_slices.append(img.astype(np.uint32))
            add_mask_vol = np.stack(add_mask_slices, axis=0)
            
            return raw_vol, seg_vol, add_mask_vol
        except Exception as e:
            print(f"Error loading volumes for {bbox_name}: {e}")
            return None, None, None

    def get_closest_component_mask(self, full_mask: np.ndarray, 
                                   z_start: int, z_end: int,
                                   y_start: int, y_end: int, 
                                   x_start: int, x_end: int,
                                   target_coord: Tuple[int, int, int]) -> np.ndarray:
        """
        Find the closest connected component in a mask to a target coordinate.
        Based on user's reference code for finding vesicle clouds closest to cleft center.
        """
        # Extract the sub-region of interest
        sub_mask = full_mask[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Label connected components in the sub-region
        labeled_sub_mask, num_features = label(sub_mask)
        
        # Return empty mask if no components found
        if num_features == 0:
            return np.zeros_like(full_mask, dtype=bool)
        
        # Find the closest component to the target coordinate
        cx, cy, cz = target_coord
        min_distance = float('inf')
        closest_label = None

        for label_num in range(1, num_features + 1):
            # Get all coordinates of pixels belonging to this component
            vesicle_coords = np.column_stack(np.where(labeled_sub_mask == label_num))

            # Calculate distances from each pixel to the target coordinate
            # Note: vesicle_coords are in (z, y, x) order from np.where
            distances = np.sqrt(
                (vesicle_coords[:, 0] + z_start - cz) ** 2 +
                (vesicle_coords[:, 1] + y_start - cy) ** 2 +
                (vesicle_coords[:, 2] + x_start - cx) ** 2
            )

            # Find the minimum distance for this component
            min_dist_for_vesicle = np.min(distances)
            
            # Update closest component if this one is closer
            if min_dist_for_vesicle < min_distance:
                min_distance = min_dist_for_vesicle
                closest_label = label_num

        # Create the output mask with only the closest component
        if closest_label is not None:
            # Create a mask for just the closest component in the sub-region
            filtered_sub_mask = (labeled_sub_mask == closest_label)
            
            # Create the full-size output mask
            combined_mask = np.zeros_like(full_mask, dtype=bool)
            combined_mask[z_start:z_end, y_start:y_end, x_start:x_end] = filtered_sub_mask
            
            return combined_mask
        else:
            # Fallback: return empty mask
            return np.zeros_like(full_mask, dtype=bool)

    def create_segment_masks(self, segmentation_volume: np.ndarray, 
                            s1_coord: Tuple[int, int, int], 
                            s2_coord: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create binary masks for the two sides from the segmentation volume."""
        x1, y1, z1 = s1_coord
        x2, y2, z2 = s2_coord
        
        try:
            seg_id_1 = segmentation_volume[z1, y1, x1]
            seg_id_2 = segmentation_volume[z2, y2, x2]
            
            mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            
            return mask_1, mask_2
        except IndexError as e:
            print(f"Error creating segment masks: {e}")
            return np.zeros_like(segmentation_volume, dtype=bool), np.zeros_like(segmentation_volume, dtype=bool)

    def extract_synapse_cube(self, 
                             raw_vol: np.ndarray,
                             seg_vol: np.ndarray, 
                             add_mask_vol: np.ndarray,
                             central_coord: Tuple[int, int, int],
                             side1_coord: Tuple[int, int, int],
                             side2_coord: Tuple[int, int, int],
                             bbox_name: str) -> Optional[torch.Tensor]:
        """
        Extract a synapse cube based on the coordinates from Excel data.
        
        Returns 3-channel tensor: [raw, presynapse_mask, cleft_mask]
        """
        # Get label mappings for this bbox
        bbox_num = bbox_name.replace("bbox", "")
        label_map = self.LABEL_MAP.get(bbox_num, 
                                      {'mito': 5, 'vesicle': 6, 'cleft': 7, 'cleft2': 7})
        
        vesicle_label = label_map['vesicle']
        cleft_label = label_map['cleft']
        cleft_label2 = label_map['cleft2']
        
        cx, cy, cz = central_coord

        # Use expanded region for vesicle analysis
        temp_half_size = self.cube_size  
        temp_x_start = max(cx - temp_half_size, 0)
        temp_x_end = min(cx + temp_half_size, raw_vol.shape[2])
        temp_y_start = max(cy - temp_half_size, 0)
        temp_y_end = min(cy + temp_half_size, raw_vol.shape[1])
        temp_z_start = max(cz - temp_half_size, 0)
        temp_z_end = min(cz + temp_half_size, raw_vol.shape[0])

        # Find vesicles in the expanded region
        vesicle_full_mask = (add_mask_vol == vesicle_label)
        temp_vesicle_mask = self.get_closest_component_mask(
            vesicle_full_mask,
            temp_z_start, temp_z_end,
            temp_y_start, temp_y_end,
            temp_x_start, temp_x_end,
            (cx, cy, cz)
        )

        # Create segment masks for both sides
        mask_1_full, mask_2_full = self.create_segment_masks(seg_vol, side1_coord, side2_coord)

        # Determine which side is the presynapse by checking overlap with vesicle mask
        overlap_side1 = np.sum(np.logical_and(mask_1_full, temp_vesicle_mask))
        overlap_side2 = np.sum(np.logical_and(mask_2_full, temp_vesicle_mask))
        presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
        
        # Get presynapse mask
        presynapse_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
        
        # Calculate the final cube bounding box
        half_size = self.cube_size // 2
        x_start = max(cx - half_size, 0)
        x_end = min(cx + half_size, raw_vol.shape[2])
        y_start = max(cy - half_size, 0)
        y_end = min(cy + half_size, raw_vol.shape[1])
        z_start = max(cz - half_size, 0)
        z_end = min(cz + half_size, raw_vol.shape[0])

        # Extract sub-volumes
        sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
        sub_presynapse_mask = presynapse_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Create cleft mask
        cleft_mask_full = ((add_mask_vol == cleft_label) | (add_mask_vol == cleft_label2))
        sub_cleft_mask = cleft_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]

        # Apply padding if needed to reach exact cube_size
        pad_z = self.cube_size - sub_raw.shape[0]
        pad_y = self.cube_size - sub_raw.shape[1]
        pad_x = self.cube_size - sub_raw.shape[2]
        
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            sub_raw = np.pad(sub_raw, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            sub_presynapse_mask = np.pad(sub_presynapse_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)
            sub_cleft_mask = np.pad(sub_cleft_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)

        # Ensure exact cube size
        sub_raw = sub_raw[:self.cube_size, :self.cube_size, :self.cube_size]
        sub_presynapse_mask = sub_presynapse_mask[:self.cube_size, :self.cube_size, :self.cube_size]
        sub_cleft_mask = sub_cleft_mask[:self.cube_size, :self.cube_size, :self.cube_size]

        # Normalize the raw volume 
        sub_raw = sub_raw.astype(np.float32)
        min_val = np.min(sub_raw)
        max_val = np.max(sub_raw)
        range_val = max_val - min_val if max_val > min_val else 1.0
        normalized_raw = (sub_raw - min_val) / range_val

        # Create the 3-channel volume
        cube = np.stack([
            normalized_raw,
            sub_presynapse_mask.astype(np.float32),
            sub_cleft_mask.astype(np.float32)
        ], axis=0)
        
        return torch.from_numpy(cube).float()

    def process_bbox_synapses(self, bbox_name: str) -> List[Tuple[str, torch.Tensor]]:
        """Process all synapses in a bbox and return a list of synapse cubes."""
        # Load volumes
        raw_vol, seg_vol, add_mask_vol = self.load_volumes(bbox_name)
        if raw_vol is None or seg_vol is None or add_mask_vol is None:
            print(f"Failed to load volumes for {bbox_name}")
            return []
        
        # Load Excel data
        excel_data = self.load_excel_data(bbox_name)
        if excel_data is None:
            print(f"Failed to load Excel data for {bbox_name}")
            return []

        synapse_cubes = []
        
        # Process each synapse in the Excel file
        for idx, row in excel_data.iterrows():
            try:
                # Extract synapse information
                synapse_id = str(row['Var1'])
                
                # Extract coordinates
                central_coord = (int(row['central_coord_1']), int(row['central_coord_2']), int(row['central_coord_3']))
                side1_coord = (int(row['side_1_coord_1']), int(row['side_1_coord_2']), int(row['side_1_coord_3']))
                side2_coord = (int(row['side_2_coord_1']), int(row['side_2_coord_2']), int(row['side_2_coord_3']))
                
                # Extract synapse cube
                cube_tensor = self.extract_synapse_cube(
                    raw_vol, seg_vol, add_mask_vol,
                    central_coord, side1_coord, side2_coord,
                    bbox_name
                )
                
                if cube_tensor is not None:
                    full_synapse_id = f"{bbox_name}_{synapse_id}"
                    synapse_cubes.append((full_synapse_id, cube_tensor))
                    print(f"Successfully extracted cube for synapse {full_synapse_id}")
                else:
                    print(f"Failed to extract cube for synapse {synapse_id} in {bbox_name}")
                    
            except (KeyError, ValueError, IndexError) as e:
                print(f"Error processing synapse {idx} in {bbox_name}: {e}")
                continue
                
        return synapse_cubes

    def process_all_synapses(self, memory_efficient: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process all synapses across all bboxes.
        
        Args:
            memory_efficient: If True, save to H5 files and return paths instead of tensors
            
        Returns:
            Dict mapping synapse_id -> cube_tensor (or H5 path if memory_efficient=True)
        """
        all_synapses = {}
        
        for bbox_name in self.bbox_names:
            print(f"Processing synapses in {bbox_name}...")
            
            # Check if H5 file exists for this bbox
            h5_file = os.path.join(self.preproc_dir, f"{bbox_name}_synapses.h5") 
            if self.create_h5 and os.path.exists(h5_file):
                try:
                    with h5py.File(h5_file, 'r') as f:
                        for synapse_id in f.keys():
                            cube = torch.from_numpy(f[synapse_id][()]).float()
                            full_id = f"{bbox_name}_{synapse_id}"
                            if memory_efficient:
                                all_synapses[full_id] = h5_file  # Store path
                            else:
                                all_synapses[full_id] = cube
                    print(f"Loaded preprocessed synapses for {bbox_name} from H5 file")
                    continue
                except Exception as e:
                    print(f"Error loading H5 file for {bbox_name}: {e}")
            
            # Process synapses for this bbox
            synapse_cubes = self.process_bbox_synapses(bbox_name)
            
            if synapse_cubes and self.create_h5:
                # Save to H5 file
                try:
                    with h5py.File(h5_file, 'w') as f:
                        for synapse_id, cube in synapse_cubes:
                            # Remove bbox prefix for storage
                            clean_id = synapse_id.replace(f"{bbox_name}_", "")
                            f.create_dataset(clean_id, data=cube.numpy().astype(np.float32))
                    print(f"Saved {len(synapse_cubes)} synapses for {bbox_name} to H5 file")
                except Exception as e:
                    print(f"Error saving H5 file for {bbox_name}: {e}")
            
            # Add to results
            for synapse_id, cube in synapse_cubes:
                if memory_efficient and self.create_h5:
                    all_synapses[synapse_id] = h5_file  # Store path
                else:
                    all_synapses[synapse_id] = cube
                    
        return all_synapses

    def load_from_paths(self, paths_dict: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Load synapse cubes from H5 file paths.
        
        Args:
            paths_dict: Dict mapping synapse_id -> H5_file_path
            
        Returns:
            Dict mapping synapse_id -> cube_tensor
        """
        result = {}
        h5_files_cache = {}  # Cache opened H5 files
        
        try:
            for synapse_id, h5_path in paths_dict.items():
                bbox_name = synapse_id.split('_')[0] + synapse_id.split('_')[1]  # e.g., bbox1
                clean_id = synapse_id.replace(f"{bbox_name}_", "")
                
                # Open H5 file if not already cached
                if h5_path not in h5_files_cache:
                    h5_files_cache[h5_path] = h5py.File(h5_path, 'r')
                
                h5_file = h5_files_cache[h5_path]
                
                if clean_id in h5_file:
                    cube = torch.from_numpy(h5_file[clean_id][()]).float()
                    result[synapse_id] = cube
                else:
                    print(f"Warning: Synapse {clean_id} not found in {h5_path}")
                    result[synapse_id] = None
                    
        finally:
            # Close all opened H5 files
            for h5_file in h5_files_cache.values():
                h5_file.close()
                
        return result


# Example usage
if __name__ == "__main__":
    loader = SynapseLoader(data_dir='data', preproc_dir='preproc_synapses', cube_size=80)
    
    # Process all synapses
    all_synapses = loader.process_all_synapses()
    
    print(f"Loaded {len(all_synapses)} synapses total")
    for synapse_id, cube in all_synapses.items():
        if cube is not None:
            print(f"  {synapse_id}: shape {cube.shape}") 