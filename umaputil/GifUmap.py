import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import random
import imageio
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageSequence
import io
import seaborn as sns
import base64  # For image encoding
import json
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch.nn.functional as F

# Add the current directory to the path so we can import the models and datasets
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import from the current project
from models.simclr3d import SimCLR
from datasets.synapse_loader import SynapseLoader
from datasets.synapse_dataset import SynapseDataset

def ensure_gif_autoplay(gif_paths, loop=0):
    """
    Ensures all GIFs are set to autoplay by modifying their loop parameter.
    
    Args:
        gif_paths: Dictionary mapping sample indices to GIF paths
        loop: Loop parameter (0 = infinite, -1 = no loop, n = number of loops)
    
    Returns:
        Dictionary with paths to modified GIFs
    """
    from PIL import Image, ImageSequence
    import os
    
    modified_gif_paths = {}
    
    for idx, path in gif_paths.items():
        try:
            # Open the original GIF
            img = Image.open(path)
            
            # Create a new file path for the modified GIF
            dir_path = os.path.dirname(path)
            file_name = os.path.basename(path)
            new_path = os.path.join(dir_path, f"autoloop_{file_name}")
            
            # Extract frames
            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            
            # Save with the loop parameter
            frames[0].save(
                new_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                loop=loop,  # 0 means infinite loop
                duration=img.info.get('duration', 100)  # Use original duration or default to 100ms
            )
            
            # Store the new path
            modified_gif_paths[idx] = new_path
            print(f"Modified GIF for sample {idx} to auto-loop")
            
        except Exception as e:
            print(f"Error modifying GIF for sample {idx}: {e}")
            # Keep the original path if modification fails
            modified_gif_paths[idx] = path
            
    return modified_gif_paths

# Define function to initialize a dataset from the current synapse project
def initialize_synapse_dataset(data_dir='datareal', preproc_dir='preproc_synapses', cube_size=80):
    """
    Initialize a SynapseDataset from the current synapse project.
    
    Args:
        data_dir: Directory containing raw data
        preproc_dir: Directory containing preprocessed H5 files
        cube_size: Size of synapse cubes
    
    Returns:
        SynapseDataset instance or None if initialization fails
    """
    try:
        print("Initializing synapse dataset...")
        
        # Initialize synapse loader
        loader = SynapseLoader(
            data_dir=data_dir,
            preproc_dir=preproc_dir,
            cube_size=cube_size
        )
        
        # Load synapses directly from H5 files
        synapses = {}
        for i in range(1, 8):
            h5_file = os.path.join(preproc_dir, f"bbox{i}_synapses.h5")
            if os.path.exists(h5_file):
                try:
                    with h5py.File(h5_file, 'r') as f:
                        for key in f.keys():
                            # Read the synapse cube
                            cube_data = f[key][:]
                            cube_tensor = torch.from_numpy(cube_data).float()
                            # Use bbox prefix for consistency
                            synapse_id = f"bbox{i}_{key}"
                            synapses[synapse_id] = cube_tensor
                    print(f"Loaded {len(list(h5py.File(h5_file, 'r').keys()))} synapses from bbox{i}")
                except Exception as e:
                    print(f"Error loading H5 file {h5_file}: {e}")
        
        if len(synapses) == 0:
            print("No synapses found in H5 files!")
            return None
            
        print(f"Total synapses loaded: {len(synapses)}")
        
        # Create dataset (no augmentation for visualization)
        dataset = SynapseDataset(
            synapse_cubes=synapses,
            augment=False
        )
        
        print(f"Successfully created dataset with {len(dataset)} samples")
        return dataset, synapses
        
    except Exception as e:
        print(f"Error initializing synapse dataset: {e}")
        return None

# Define the perform_clustering_analysis function here instead of importing it
def perform_clustering_analysis(features_df, n_clusters=10, output_path=None):
    """
    Perform clustering analysis on features DataFrame.
    
    Args:
        features_df: DataFrame containing features
        n_clusters: Number of clusters for K-means
        output_path: Directory to save clustering results
    
    Returns:
        features_df: DataFrame with cluster assignments
    """
    print(f"Starting clustering analysis with {n_clusters} clusters")
    
    if output_path:
    # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # Find feature columns
    feature_cols = [col for col in features_df.columns if col.startswith('feat_') or 'embedding' in col]
    if not feature_cols:
        # Try other common feature column patterns
        feature_cols = [col for col in features_df.columns if col not in ['cluster', 'x', 'y', 'bbox_name']]
        
    if not feature_cols:
        print("Error: No feature columns found in DataFrame")
        return features_df
        
    print(f"Using {len(feature_cols)} feature columns for clustering")
    
    # Extract features and standardize
    features = features_df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to DataFrame
    features_df = features_df.copy()
    features_df['cluster'] = cluster_labels
    
    # Calculate silhouette score
    sil_score = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette score: {sil_score:.3f}")
    
    if output_path:
    # Save clustered features
        features_df.to_csv(output_dir / "clustered_features.csv", index=False)
    
    # Define color mapping for different bounding boxes
    color_mapping = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
    return features_df

def create_gif_from_volume(volume, output_path, fps=10, segmentation_type=None):
    """
    Create a GIF from a volume (3D array) and return the frames.
    
    Args:
        volume: 3D array representing volume data
        output_path: Path to save the GIF
        fps: Frames per second
        segmentation_type: Type of segmentation used - if type 13, only show center 25 frames
        
    Returns:
        Tuple of (output_path, frames) where frames is a list of frame data for web display
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().detach().numpy()
    
    # Ensure volume is a numpy array
    if not isinstance(volume, np.ndarray):
        raise TypeError(f"Volume must be a numpy array or PyTorch tensor, got {type(volume)}")
    
    # Handle different volume shapes
    print(f"Original volume shape: {volume.shape}")
    
    # SynapseDataset returns 4D tensors: (channels, depth, height, width)
    # We want the raw EM channel (channel 0) for visualization
    if volume.ndim == 4:
        # Take the first channel (raw EM data)
        volume = volume[0]  # Now shape is (depth, height, width)
        print(f"Using channel 0, new shape: {volume.shape}")
    elif volume.ndim == 3:
        # Already 3D, assume it's (depth, height, width)
        print(f"Volume is already 3D: {volume.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D with shape {volume.shape}")
    
    # For segmentation type 13, only show the center 25 frames (27-53)
    if segmentation_type == 13 and volume.shape[0] >= 54:
        print(f"Segmentation type 13 detected: Using only center frames 27-53 (25 frames)")
        volume = volume[27:54]  # Python indexing is 0-based, so 27-53 is 27:54
    
    # Prepare frames for GIF
    frames = []
    
    # Use absolute fixed scaling to match dataloader3.py behavior
    # This ensures completely consistent gray values across all samples
    
    # Define same fixed values as in dataloader3.py
    fixed_min = 0.0
    fixed_max = 255.0
    
    # If values are in 0-1 range, scale to 0-255 for processing
    if volume.max() <= 1.0:
        volume = volume * 255.0
        
    print(f"Using ABSOLUTE fixed gray values: min={fixed_min}, max={fixed_max}")
    print(f"Volume range before clipping: {volume.min():.4f}-{volume.max():.4f}")
        
    for i in range(volume.shape[0]):
        frame = volume[i]
        
        # Frame should be 2D now since we took volume[0] if it was 4D
        if frame.ndim != 2:
            raise ValueError(f"Frame {i} should be 2D but has shape {frame.shape}")
            
        # Clip to fixed range without any normalization
        clipped = np.clip(frame, fixed_min, fixed_max)
        # Convert to uint8 for GIF
        scaled = clipped.astype(np.uint8)
        frames.append(scaled)
    
    if not frames:
        raise ValueError("No valid frames were generated from the volume")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Creating GIF with {len(frames)} frames, each {frames[0].shape}")
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)
    
    # Convert frames to base64-encoded PNGs for web display
    frame_data = []
    for frame in frames:
        # Convert frame to PNG and then to base64
        with io.BytesIO() as output:
            Image.fromarray(frame).save(output, format="PNG")
            frame_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
            frame_data.append(frame_base64)
    
    return output_path, frame_data


def create_animated_gif_visualization(features_df, gif_paths, output_dir, dim_reduction='umap', frame_data=None, max_slices_data=None):
    """
    Create a simple HTML page that displays animated GIFs directly at their coordinates.
    The GIFs are embedded directly in the HTML as base64 data to avoid file:// protocol issues.
    GIFs are made draggable so users can rearrange them.
    Includes a control to adjust how many GIFs are displayed at runtime.
    
    Args:
        features_df: DataFrame with features and coordinates
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Directory to save the HTML file
        dim_reduction: Dimensionality reduction method ('umap')
        frame_data: Dictionary mapping sample indices to lists of frame data (base64 encoded images)
        max_slices_data: Dictionary mapping sample indices to max slice information
    
    Returns:
        Path to the HTML file
    """
    method_name = "UMAP" if dim_reduction == 'umap' else "t-SNE"
    import base64
    
    # Define plot dimensions upfront
    plot_width = 1600  # From the CSS .plot-container width
    plot_height = 1200  # From the CSS .plot-container height
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Ensuring coordinates are available...")
    
    # Make sure we have the right coordinate columns
    coord_columns = {
        'umap': ['umap_x', 'umap_y'],
        'generic': ['x', 'y']
    }
    
    # First priority: check for generic columns
    if all(col in features_df.columns for col in coord_columns['generic']):
        x_col, y_col = coord_columns['generic']
        print(f"Using generic coordinate columns: {x_col}, {y_col}")
    # Second priority: check for method-specific columns
    elif all(col in features_df.columns for col in coord_columns[dim_reduction]):
        x_col, y_col = coord_columns[dim_reduction]
        print(f"Using {method_name}-specific coordinate columns: {x_col}, {y_col}")
    # Fall back to the other method if available
    else:
        raise ValueError(f"No suitable coordinate columns found in DataFrame. Available columns: {features_df.columns.tolist()}")
    
    # Extract coordinates and other info for samples with GIFs
    samples_with_gifs = []
    
    # Track how many GIFs we have from each cluster for reporting
    cluster_counts = {}
    
    for idx in gif_paths:
        if idx in features_df.index:
            sample = features_df.loc[idx]
            
            # Use the determined coordinate columns
            if x_col in sample and y_col in sample:
                x, y = sample[x_col], sample[y_col]
                
                # Extract cluster and bbox information if available
                cluster = sample.get('cluster', 'N/A') if 'cluster' in sample else 'N/A'
                bbox = sample.get('bbox_name', 'unknown') if 'bbox_name' in sample else 'unknown'
                
                # Extract central coordinates if available
                central_coord_1 = sample.get('central_coord_1', 0) if 'central_coord_1' in sample else 0
                central_coord_2 = sample.get('central_coord_2', 0) if 'central_coord_2' in sample else 0
                central_coord_3 = sample.get('central_coord_3', 0) if 'central_coord_3' in sample else 0
                
                # Count samples per cluster
                if cluster != 'N/A':
                    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                
                # Convert numpy/pandas types to Python native types for JSON serialization
                if hasattr(idx, 'item'):
                    idx = idx.item()
                if hasattr(x, 'item'):
                    x = x.item()
                if hasattr(y, 'item'):
                    y = y.item()
                if hasattr(cluster, 'item'):
                    cluster = cluster.item()
                
                # Load the GIF file and convert to base64
                try:
                    with open(gif_paths[idx], 'rb') as gif_file:
                        gif_data = gif_file.read()
                        encoded_gif = base64.b64encode(gif_data).decode('utf-8')
                        
                        # Add frame data if available
                        frames = []
                        if frame_data and idx in frame_data:
                            frames = frame_data[idx]
                        
                        # Add max slices if available
                        max_slices = None
                        if max_slices_data and idx in max_slices_data:
                            max_slices = max_slices_data[idx]
                        
                        samples_with_gifs.append({
                            'id': idx,
                            'x': x,
                            'y': y,
                            'cluster': cluster,
                            'bbox': bbox,
                            'central_coord_1': central_coord_1,
                            'central_coord_2': central_coord_2, 
                            'central_coord_3': central_coord_3,
                            'gifData': encoded_gif,
                            'frames': frames,
                            'max_slices': max_slices
                        })
                except Exception as e:
                    print(f"Error encoding GIF for sample {idx}: {e}")
            else:
                print(f"Warning: Sample {idx} does not have required coordinates ({x_col}, {y_col}). Skipping.")
    
    # Print distribution of GIFs across clusters
    if cluster_counts:
        print("\nDistribution of GIFs across clusters:")
        for cluster, count in sorted(cluster_counts.items()):
            print(f"  Cluster {cluster}: {count} GIFs")
    
    if not samples_with_gifs:
        raise ValueError("No samples with GIFs and valid coordinates found. Cannot create visualization.")
    
    print(f"\nTotal samples with GIFs and valid coordinates: {len(samples_with_gifs)}")
    print(f"First sample for debugging: {json.dumps(samples_with_gifs[0], default=str)[:200]}...")
    
    # Compute the bounds of the coordinate values
    all_x_values = features_df[x_col].values
    all_y_values = features_df[y_col].values
    
    print(f"X coordinate range: {min(all_x_values)} to {max(all_x_values)}")
    print(f"Y coordinate range: {min(all_y_values)} to {max(all_y_values)}")
    
    x_min, x_max = float(min(all_x_values)), float(max(all_x_values))
    y_min, y_max = float(min(all_y_values)), float(max(all_y_values))
    
    # Add padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    
    # Processing to create non-overlapping positions
    gif_size = 50  # Default size decreased from 100 to 50px
    shift_limit = 75  # Increased from 50 to 100px shift limit
    max_shift_x = shift_limit
    max_shift_y = shift_limit
    
    # Function to check if two rectangles overlap
    def do_rectangles_overlap(rect1, rect2):
        return not (rect1['right'] < rect2['left'] or 
                   rect1['left'] > rect2['right'] or 
                   rect1['bottom'] < rect2['top'] or 
                   rect1['top'] > rect2['bottom'])
    
    # Track placed rectangles to avoid overlap
    placed_rectangles = []
    
    # Function to find non-overlapping position
    def find_non_overlapping_position(baseX, baseY, existingRects):
        # Check if the original position works
        half_size = gif_size / 2
        rect = {
            'left': baseX - half_size,
            'right': baseX + half_size,
            'top': baseY - half_size,
            'bottom': baseY + half_size
        }
        
        # Check if original position has no overlap
        has_overlap = False
        overlap_rect = None
        
        for existing_rect in existingRects:
            if do_rectangles_overlap(rect, existing_rect):
                has_overlap = True
                overlap_rect = existing_rect
                break
                
        # If no overlap, use original position
        if not has_overlap:
            return (baseX, baseY, rect)
            
        # Calculate the minimum shift needed in each direction to avoid overlap
        if overlap_rect:
            # Calculate overlap amounts in each direction
            overlap_right = rect['right'] - overlap_rect['left']
            overlap_left = overlap_rect['right'] - rect['left']
            overlap_bottom = rect['bottom'] - overlap_rect['top']
            overlap_top = overlap_rect['bottom'] - rect['top']
            
            # Find the smallest shift needed
            shifts = [
                {'axis': 'x', 'amount': overlap_right, 'direction': 1},   # shift right
                {'axis': 'x', 'amount': -overlap_left, 'direction': -1},  # shift left
                {'axis': 'y', 'amount': overlap_bottom, 'direction': 1},  # shift down
                {'axis': 'y', 'amount': -overlap_top, 'direction': -1}    # shift up
            ]
            
            # Sort by absolute amount to find smallest shift
            shifts.sort(key=lambda s: abs(s['amount']))
            
            # Try each shift until we find one that works
            for shift in shifts:
                # Skip if shift is too large
                if abs(shift['amount']) > shift_limit:
                    continue
                
                shifted_x = baseX
                shifted_y = baseY
                
                if shift['axis'] == 'x':
                    shifted_x += shift['amount']
                else:
                    shifted_y += shift['amount']
                
                # Skip if this would move the GIF out of bounds
                if (shifted_x - half_size < 0 or shifted_x + half_size > plot_width or
                    shifted_y - half_size < 0 or shifted_y + half_size > plot_height):
                    continue
                
                # Check if this position works with all existing rectangles
                shifted_rect = {
                    'left': shifted_x - half_size,
                    'right': shifted_x + half_size,
                    'top': shifted_y - half_size,
                    'bottom': shifted_y + half_size
                }
                
                shifted_overlap = False
                for existing_rect in existingRects:
                    if do_rectangles_overlap(shifted_rect, existing_rect):
                        shifted_overlap = True
                        break
                        
                if not shifted_overlap:
                    return (shifted_x, shifted_y, shifted_rect)
        
        # If the simple shifts didn't work, try a more general approach
        # Try cardinal and diagonal directions with increasing distances
        directions = [
            (1, 0),   # right
            (0, 1),   # down
            (-1, 0),  # left
            (0, -1),  # up
            (1, 1),   # down-right
            (1, -1),  # up-right
            (-1, 1),  # down-left
            (-1, -1)  # up-left
        ]
        
        # Try increasing distances with smaller steps
        for distance in range(1, int(shift_limit) + 1):
            for dir_x, dir_y in directions:
                shifted_x = baseX + (dir_x * distance)
                shifted_y = baseY + (dir_y * distance)
                
                # Skip if this would move the GIF out of bounds
                if (shifted_x - half_size < 0 or shifted_x + half_size > plot_width or
                    shifted_y - half_size < 0 or shifted_y + half_size > plot_height):
                    continue
                
                # Check this position
                shifted_rect = {
                    'left': shifted_x - half_size,
                    'right': shifted_x + half_size,
                    'top': shifted_y - half_size,
                    'bottom': shifted_y + half_size
                }
                
                shifted_overlap = False
                for existing_rect in existingRects:
                    if do_rectangles_overlap(shifted_rect, existing_rect):
                        shifted_overlap = True
                        break
                        
                if not shifted_overlap:
                    return (shifted_x, shifted_y, shifted_rect)
        
        # Try with slightly smaller GIF size as a last resort
        reduced_half_size = half_size * 0.8
        for distance in range(1, int(shift_limit) + 1, 2):
            for dir_x, dir_y in directions:
                shifted_x = baseX + (dir_x * distance)
                shifted_y = baseY + (dir_y * distance)
                
                # Skip if this would move the GIF out of bounds
                if (shifted_x - reduced_half_size < 0 or shifted_x + reduced_half_size > plot_width or
                    shifted_y - reduced_half_size < 0 or shifted_y + reduced_half_size > plot_height):
                    continue
                
                # Check this position with reduced size
                shifted_rect = {
                    'left': shifted_x - reduced_half_size,
                    'right': shifted_x + reduced_half_size,
                    'top': shifted_y - reduced_half_size,
                    'bottom': shifted_y + reduced_half_size
                }
                
                shifted_overlap = False
                for existing_rect in existingRects:
                    if do_rectangles_overlap(shifted_rect, existing_rect):
                        shifted_overlap = True
                        break
                        
                if not shifted_overlap:
                    return (shifted_x, shifted_y, shifted_rect)
        
        # If we can't find a non-overlapping position, return null
        return None
    
    # Initialize originalPositions dictionary for storing initial GIF positions
    # Note: we need to map the raw coordinates to plot coordinates
    # For this we use the same mapping logic as in the JavaScript mapToPlot function
    orig_positions_dict = {}
    samples_to_remove = []
    
    for i, sample in enumerate(samples_with_gifs):
        id_val = sample['id']
        if hasattr(id_val, 'item'):
            id_val = id_val.item()  # Convert numpy types to native Python
        
        x_val = sample['x']
        y_val = sample['y']
        
        # Map data coordinates to plot coordinates (same as JavaScript mapToPlot function)
        plot_x = ((x_val - x_min) / (x_max - x_min)) * plot_width
        # Invert y-axis (data coordinates increase upward, plot coordinates increase downward)
        plot_y = plot_height - ((y_val - y_min) / (y_max - y_min)) * plot_height
        
        # Find non-overlapping position
        position = find_non_overlapping_position(plot_x, plot_y, placed_rectangles)
        
        # If no valid position found, skip this sample
        if position is None:
            print(f"Skipping sample {id_val} due to overlap that couldn't be resolved")
            samples_to_remove.append(i)
            continue
            
        # Unpack the position
        pos_x, pos_y, rect = position
        
        # Add to tracking for future samples
        placed_rectangles.append(rect)
        
        # Use string keys for the JavaScript object
        str_id = str(id_val)
        orig_positions_dict[str_id] = {"x": float(pos_x), "y": float(pos_y)}
        
        # If position was shifted, note it
        if pos_x != plot_x or pos_y != plot_y:
            print(f"Sample {id_val} shifted to avoid overlap")
    
    # Remove samples that couldn't be placed
    if samples_to_remove:
        for i in sorted(samples_to_remove, reverse=True):
            del samples_with_gifs[i]
        print(f"Removed {len(samples_to_remove)} samples that couldn't be placed without overlap")
    
    # Convert to JSON string for embedding in JavaScript
    originalPositions = json.dumps(orig_positions_dict)
    print(f"originalPositions JSON string length: {len(originalPositions)}")
    print(f"Sample of originalPositions: {originalPositions[:100]}...")
    
    # Define colors for points based on clusters or bboxes
    point_colors = {}
    bbox_colors = {}
    
    if 'cluster' in features_df.columns:
        # Generate colors for each cluster
        clusters = features_df['cluster'].unique()
        import matplotlib.pyplot as plt
        
        # Use the new recommended approach to get colormaps
        try:
            cmap = plt.colormaps['tab10']
        except AttributeError:
            # Fallback for older matplotlib versions
            cmap = plt.cm.get_cmap('tab10')
        
        for i, cluster in enumerate(clusters):
            r, g, b, _ = cmap(i % 10)  # Use modulo to handle more than 10 clusters
            point_colors[cluster] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    
    # Generate colors for bboxes
    if 'bbox_name' in features_df.columns:
        bboxes = features_df['bbox_name'].unique()
        bbox_colors_list = [
            '#FF0000', '#00FFFF', '#FFA500', '#800080', 
            '#008000', '#0000FF', '#FF00FF', '#FFFF00', 
            '#808080', '#000000'
        ]
        
        for i, bbox in enumerate(bboxes):
            if i < len(bbox_colors_list):
                bbox_colors[bbox] = bbox_colors_list[i]
            else:
                # Generate a random color if we run out of predefined colors
                r = random.randint(0, 255)
                g = random.randint(0, 255) 
                b = random.randint(0, 255)
                bbox_colors[bbox] = f"rgb({r}, {g}, {b})"
    
    # -------------------------------------------------------------------
    # Pre-compute WebKnossos URLs (needs samples_with_gifs)
    # -------------------------------------------------------------------
    wk_urls = {}
    try:
        from umaputil.WebknossosUrl import calculate_bbox_coordinates as _calc_wk
    except Exception:
        _calc_wk = None

    if _calc_wk is not None:
        for sample in samples_with_gifs:
            bbox_name = sample.get('bbox', 'unknown')
            if isinstance(bbox_name, str) and bbox_name.startswith('bbox'):
                bb_num = int(bbox_name.replace('bbox', ''))
                cx = int(sample.get('central_coord_1', 40))
                cy = int(sample.get('central_coord_2', 40))
                cz = int(sample.get('central_coord_3', 40))
                try:
                    _x1, _y1, _z1 = _calc_wk(cx, cy, cz, bb_num)
                    wk_urls[sample['id']] = f"https://webknossos.brain.mpg.de/annotations/67bcfa0301000006202da79c#{_x1},{_y1},{_z1},0,0.905,1506"
                except Exception as e:
                    print(f"WK URL calculation failed for sample {sample['id']}: {e}")

    # Generate HTML content for data points
    points_content = ""
    for idx, row in features_df.iterrows():
        if x_col in row and y_col in row:
            x, y = row[x_col], row[y_col]
            
            # Convert to native Python types
            if hasattr(idx, 'item'):
                idx = idx.item()
            if hasattr(x, 'item'):
                x = x.item()
            if hasattr(y, 'item'):
                y = y.item()
            
            # Determine color based on cluster
            cluster_color = 'rgb(100, 100, 100)'
            cluster = None
            if 'cluster' in row:
                cluster = row['cluster']
                if hasattr(cluster, 'item'):
                    cluster = cluster.item()
                cluster_color = point_colors.get(cluster, 'rgb(100, 100, 100)')
            
            # Get bbox_name and color based on bbox
            bbox_name = row.get('bbox_name', 'unknown')
            if hasattr(bbox_name, 'item'):
                bbox_name = str(bbox_name.item())
            else:
                bbox_name = str(bbox_name)
            
            bbox_color = bbox_colors.get(bbox_name, 'rgb(100, 100, 100)')
            
            # Get Var1 for tooltip
            var1 = row.get('Var1', f'sample_{idx}')
            if hasattr(var1, 'item'):
                var1 = str(var1.item())
            else:
                var1 = str(var1)
            
            # Add this point to the samples array - make sure we have a valid number before adding
            if not (np.isnan(x) or np.isnan(y)):
                points_content += f"""
                {{
                    "id": {idx},
                    "x": {x},
                    "y": {y},
                    "color": "{cluster_color}",
                    "bbox_color": "{bbox_color}",
                    "cluster": "{str(cluster) if cluster is not None else 'unknown'}",
                    "hasGif": {str(idx in gif_paths).lower()},
                    "bbox_name": "{bbox_name}",
                    "var1": "{var1}"
                }},"""
    
    # Count how many valid points we have
    print(f"Generated points_content with {points_content.count('id:')} points")
    print(f"Sample of points_content: {points_content[:200]}...")
    
    # Generate HTML content for GIFs
    gifs_content = ""
    for sample in samples_with_gifs:
        # Only include frames data if we have it
        has_frames = sample.get('frames') is not None and len(sample.get('frames', [])) > 0
        
        # Include max_slices data if available
        max_slices_json = 'null'
        if sample.get('max_slices') is not None:
            try:
                # Convert numpy int64 values to Python integers for JSON serialization
                max_slices = sample.get('max_slices')
                cleaned_max_slices = {}
                for key, value in max_slices.items():
                    if hasattr(value, 'item'):  # Check if it's a numpy scalar
                        cleaned_max_slices[key] = value.item()  # Convert to Python scalar
                    else:
                        cleaned_max_slices[key] = value
                
                max_slices_json = json.dumps(cleaned_max_slices)
                print(f"Successfully serialized max_slices for sample {sample.get('id')}: {max_slices_json}")
            except Exception as e:
                print(f"Error serializing max_slices for sample {sample.get('id')}: {e}")
                max_slices_json = 'null'
        
        gifs_content += f"""{{
            "id": {sample.get('id', 0)},
            "x": {sample.get('x', 0)},
            "y": {sample.get('y', 0)},
            "cluster": "{sample.get('cluster', 'N/A')}",
            "bbox": "{sample.get('bbox', 'unknown')}",
            "central_coord_1": {sample.get('central_coord_1', 0)},
            "central_coord_2": {sample.get('central_coord_2', 0)},
            "central_coord_3": {sample.get('central_coord_3', 0)},
            "gifData": "{sample['gifData']}",
            "hasFrames": {str(has_frames).lower()},
            "max_slices": {max_slices_json},
            "wk_url": "{wk_urls.get(sample.get('id'), '')}"
        }},"""
    
    # Count how many valid GIFs we have
    print(f"Generated gifs_content with {gifs_content.count('id:')} GIFs")
    print(f"Sample of gifs_content (without actual base64 data): {gifs_content[:200]}...")
    
    # Create a dedicated frames content structure
    frames_content = "{"
    has_any_frames = False
    
    # Check if we have frame data
    if frame_data:
        for idx, frames in frame_data.items():
            if frames:
                has_any_frames = True
                # Stringify the ID
                str_id = str(idx)
                if hasattr(idx, 'item'):
                    str_id = str(idx.item())
                
                # Add frames data for this sample as JSON array
                frames_content += f'"{str_id}": ['
                for frame in frames:
                    frames_content += f'"{frame}",'
                # Remove trailing comma if there are frames
                if frames:
                    frames_content = frames_content[:-1]
                frames_content += "],"
    
    # Remove trailing comma if any frames were added
    if frames_content.endswith(","):
        frames_content = frames_content[:-1]
    
    frames_content += "}"
    
    # If we have no frames, initialize a valid empty object
    if not has_any_frames:
        frames_content = "{}"
        
    print(f"Generated frames_content with data for {frame_data.keys() if frame_data else 0} GIFs")
    print(f"Has any frames: {has_any_frames}")
    print(f"frames_content length: {len(frames_content)}")
    
    # -------------------------------------------------------------------
    # Ensure bbox_name and central coordinates columns exist early
    # -------------------------------------------------------------------
    if 'bbox_name' not in features_df.columns:
        import re
        def _derive_bbox(name):
            m = re.match(r'(bbox\d+)', str(name))
            return m.group(1) if m else 'unknown'
        if 'synapse_name' in features_df.columns:
            features_df['bbox_name'] = features_df['synapse_name'].apply(_derive_bbox)
        elif 'Var1' in features_df.columns:
            features_df['bbox_name'] = features_df['Var1'].apply(_derive_bbox)
        else:
            features_df['bbox_name'] = features_df.index.map(lambda x: _derive_bbox(x))

    for col in ['central_coord_1', 'central_coord_2', 'central_coord_3']:
        if col not in features_df.columns:
            features_df[col] = 40  # default cube center

    # Read the HTML template
    template_path = os.path.join(os.path.dirname(__file__), "template.html")
    try:
        with open(template_path, 'r', encoding='utf-8') as template_file:
            html_content = template_file.read()
            
        # Replace placeholders with actual data
        # Remove trailing commas from content to prevent JavaScript syntax errors
        points_content_clean = points_content.rstrip(',').rstrip()
        gifs_content_clean = gifs_content.rstrip(',').rstrip()
        
        html_content = html_content.replace('{method_name}', method_name)
        html_content = html_content.replace('{x_min}', str(x_min))
        html_content = html_content.replace('{x_max}', str(x_max))
        html_content = html_content.replace('{y_min}', str(y_min))
        html_content = html_content.replace('{y_max}', str(y_max))
        html_content = html_content.replace('{originalPositions}', originalPositions)
        html_content = html_content.replace('{frames_content}', frames_content)
        html_content = html_content.replace('{points_content}', points_content_clean)
        html_content = html_content.replace('{gifs_content}', gifs_content_clean)
        html_content = html_content.replace('{len(samples_with_gifs)}', str(len(samples_with_gifs)))
        html_content = html_content.replace('{segmentation_type}', '\'none\'')  # Default to 'none' since we don't have segmentation
        
        print("Successfully loaded and processed HTML template")
    except Exception as e:
        print(f"Error loading or processing HTML template: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save the HTML file
    html_path = output_dir / f"animated_gifs_{dim_reduction}_visualization.html"
    try:
        # # Add debug statements to narrow down the error
        # print("Types of data:")
        # print("- originalPositions type:", type(originalPositions))
        # print("- frames_content type:", type(frames_content))
        # print("- frames_content length:", len(frames_content))
        # print("- First item in samples_with_gifs:", samples_with_gifs[0] if samples_with_gifs else "No samples")
        
        # # Check if the points_content and gifs_content are empty 
        # if not points_content.strip():
        #     print("WARNING: points_content is empty! No background points will be shown.")
        
        # if not gifs_content.strip():
        #     print("WARNING: gifs_content is empty! No GIFs will be shown.")
            
        # Write the HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # print(f"Created animated GIF visualization with embedded data: {html_path}")
        
        # Print a sample of the HTML content to verify it has data
        html_sample = html_content[0:1000]  # Get first 1000 chars
        # print(f"Sample of HTML content: {html_sample}")
        
        # Check HTML file size
        html_size = os.path.getsize(html_path)
        # print(f"HTML file size: {html_size} bytes")
        
    except Exception as e:
        print(f"Error creating animated GIF visualization: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
    
    return html_path

def extract_max_slices_simple(volume):
    """Find slice indices where the cleft mask (channel-2) has the most voxels.

    Returns a dict: {'xy': z_idx, 'xz': y_idx, 'yz': x_idx}
    """
    try:
        # Convert to numpy if it's a tensor
        if torch.is_tensor(volume):
            volume = volume.detach().cpu().numpy()

        # Ensure 4-D (C, D, H, W)
        if volume.ndim == 3:
            # Assume missing channel dim – treat raw only, so no mask available
            D, H, W = volume.shape
            mask = np.zeros((D, H, W), dtype=bool)
        else:
            mask = volume[2] > 0  # cleft mask channel

        # If mask empty, fallback to centre slice
        if mask.sum() == 0:
            D, H, W = mask.shape
            return {'xy': D // 2, 'xz': H // 2, 'yz': W // 2}

        # Sum along axes
        z_sums = mask.sum(axis=(1, 2))  # iterate over z (xy view)
        y_sums = mask.sum(axis=(0, 2))  # iterate over y (xz view)
        x_sums = mask.sum(axis=(0, 1))  # iterate over x (yz view)

        return {
            'xy': int(z_sums.argmax()),
            'xz': int(y_sums.argmax()),
            'yz': int(x_sums.argmax())
        }
    except Exception as e:
        print(f"Error extracting max cleft slices: {e}")
        # Fallback: centre slice
        if torch.is_tensor(volume):
            _, D, H, W = volume.shape
        else:
            D, H, W = volume.shape[-3:]
        return {'xy': D // 2, 'xz': H // 2, 'yz': W // 2}

def extract_embeddings_from_checkpoint(checkpoint_path, data_dir='datareal', preproc_dir='preproc_synapses', 
                                     cube_size=80, max_samples=1000, hidden_dim=512, output_dim=128):
    """
    Extract embeddings directly from a trained SimCLR checkpoint.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        data_dir: Directory containing raw data
        preproc_dir: Directory containing preprocessed H5 files  
        cube_size: Size of synapse cubes
        max_samples: Maximum number of samples to process
        hidden_dim: Hidden dimension of the model
        output_dim: Output dimension of the projection head
        
    Returns:
        features_df: DataFrame with embeddings and metadata
        dataset: The dataset used for extraction
        synapses: Dictionary of synapse data
    """
    print(f"Extracting embeddings from checkpoint: {checkpoint_path}")
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = SimCLR(in_channels=3, hidden_dim=hidden_dim, out_dim=output_dim)
    
    # Load checkpoint with fallback for PyTorch 2.6
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Warning: safe checkpoint load failed ({e}). Retrying with weights_only=False...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state dict with fallback
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded full model state dict")
    except Exception as e:
        print(f"Failed to load full model state dict: {e}")
        print("Trying to load only backbone...")
        # Extract only backbone weights
        backbone_state = {k.replace('backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                         if k.startswith('backbone.')}
        model.backbone.load_state_dict(backbone_state)
        print("Loaded backbone state dict")
    
    model.to(device)
    model.eval()
    
    # Initialize dataset
    print("Initializing dataset...")
    result = initialize_synapse_dataset(data_dir=data_dir, preproc_dir=preproc_dir, cube_size=cube_size)
    if result is None:
        raise ValueError("Failed to initialize dataset")
    
    dataset, synapses = result
    print(f"Dataset initialized with {len(dataset)} samples")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = []
    synapse_names = []
    bbox_names = []
    
    # Limit samples if requested
    num_samples = min(max_samples, len(dataset))
    indices = list(range(num_samples))
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = []
            batch_names = []
            batch_bboxes = []
            
            for idx in batch_indices:
                try:
                    sample = dataset[idx]
                    if isinstance(sample, tuple):
                        volume = sample[0]
                    else:
                        volume = sample

                    # True synapse identifier from dataset
                    try:
                        synapse_name = dataset.synapse_ids[idx]
                    except Exception:
                        synapse_name = f"sample_{idx}"
                    
                    # Extract bbox name from synapse name
                    if 'bbox' in synapse_name:
                        bbox_name = synapse_name.split('_')[0]  # Extract bbox1, bbox2, etc.
                    else:
                        bbox_name = 'unknown'
                    
                    batch_data.append(volume)
                    batch_names.append(synapse_name)
                    batch_bboxes.append(bbox_name)
                    
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue
            
            if batch_data:
                # Stack batch
                batch_tensor = torch.stack(batch_data).to(device)
                
                # Get embeddings (use backbone only, not projection head)
                with torch.no_grad():
                    batch_embeddings = model.backbone(batch_tensor)
                    # Global average pooling if needed
                    if len(batch_embeddings.shape) > 2:
                        batch_embeddings = F.adaptive_avg_pool3d(batch_embeddings, (1, 1, 1))
                        batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                synapse_names.extend(batch_names)
                bbox_names.extend(batch_bboxes)
                
                if len(synapse_names) % 100 == 0:
                    print(f"Processed {len(synapse_names)} samples...")
    
    if not embeddings:
        raise ValueError("No embeddings extracted")
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings, axis=0)
    print(f"Extracted embeddings shape: {all_embeddings.shape}")
    
    # Create DataFrame
    embedding_cols = [f'feat_{i}' for i in range(all_embeddings.shape[1])]
    features_df = pd.DataFrame(all_embeddings, columns=embedding_cols)
    features_df['synapse_name'] = synapse_names
    features_df['bbox_name'] = bbox_names
    features_df['Var1'] = synapse_names  # For compatibility with visualization
    
    print(f"Created features DataFrame with {len(features_df)} samples and {len(embedding_cols)} features")
    
    return features_df, dataset, synapses

def main():
    """Main function for running visualization from checkpoint."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create UMAP/t-SNE visualization with GIFs from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained SimCLR checkpoint')
    parser.add_argument('--data_dir', type=str, default='datareal',
                        help='Directory containing raw data')
    parser.add_argument('--preproc_dir', type=str, default='preproc_synapses',
                        help='Directory containing preprocessed H5 files')
    parser.add_argument('--output_dir', type=str, default='umap_visualization',
                        help='Output directory for visualization')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Maximum number of samples to process for embeddings')
    parser.add_argument('--num_samples', type=int, default=500,
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found")
        return
        
    # Extract embeddings directly from checkpoint
    try:
        features_df, dataset, synapses = extract_embeddings_from_checkpoint(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            preproc_dir=args.preproc_dir,
            cube_size=args.cube_size,
            max_samples=args.max_samples,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim
        )
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return
    
    # Perform clustering if not already done
    if 'cluster' not in features_df.columns:
        print("No cluster column found, performing clustering...")
        features_df = perform_clustering_analysis(features_df, n_clusters=args.n_clusters, output_path=args.output_dir)
    
    # Create visualization with selected dim reduction method
    print(f"Creating {args.dim_reduction.upper()} visualization with GIFs for {args.num_samples} random samples...")
    
    # Check dataset length to ensure we only choose valid indices
    try:
        dataset_length = len(dataset)
        print(f"Dataset contains {dataset_length} samples")
        
        # Make sure the dataset has samples
        if dataset_length == 0:
            raise ValueError("Dataset is empty")
            
        # Get valid indices that are both in features_df and within dataset range
        valid_indices = [i for i in features_df.index if i < dataset_length]
        if len(valid_indices) == 0:
            print("Warning: No valid indices found that exist in both the dataset and features DataFrame.")
            print("Creating visualization without sample GIFs.")
            valid_indices = []
    except Exception as e:
        print(f"Warning: Could not determine dataset length: {e}")
        print("Assuming all feature indices are valid.")
        valid_indices = features_df.index.tolist()
    
    # Compute coordinates if not already in the DataFrame
    if 'x' not in features_df.columns or 'y' not in features_df.columns:
        print(f"Computing {args.dim_reduction.upper()} coordinates...")
        feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
        if not feature_cols:
            feature_cols = [col for col in features_df.columns if 'layer' in col and 'feat_' in col]
            
        if feature_cols:
            features = features_df[feature_cols].values
            features_scaled = StandardScaler().fit_transform(features)
            
            if args.dim_reduction == 'umap':
                # Use UMAP
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                results = reducer.fit_transform(features_scaled)
            elif args.dim_reduction == 'tsne':
                # Use t-SNE
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                results = reducer.fit_transform(features_scaled)
            else:
                raise ValueError(f"Unsupported dimensionality reduction method: {args.dim_reduction}")
            
            features_df['x'] = results[:, 0]
            features_df['y'] = results[:, 1]
            
            # Save the updated features with coordinates
            output_csv = Path(args.output_dir) / "clustered_features.csv"
            features_df.to_csv(output_csv, index=False)
            print(f"Saved features with {args.dim_reduction.upper()} coordinates to {output_csv}")
            
        else:
            print("ERROR: No feature columns found in the DataFrame")
            return

    # Select random samples for GIFs
    # If there are valid indices, select samples and create GIFs
    if valid_indices:
        # Create output directories
        gifs_dir = output_dir / "sample_gifs"
        gifs_dir.mkdir(parents=True, exist_ok=True)
        
        # Select random samples, making sure they cover different clusters if possible
        np.random.seed(42)
        random_samples = []
        
        # Use the number of samples requested by user
        num_samples = args.num_samples

        if 'cluster' in features_df.columns:
            # Get all cluster IDs
            clusters = features_df['cluster'].unique()
            
            # Get approximately even samples from each cluster (limited to valid indices)
            samples_per_cluster = max(1, num_samples // len(clusters))
            remaining_samples = num_samples - (samples_per_cluster * len(clusters))
            
            for cluster in clusters:
                # Get samples that are both in this cluster AND in valid_indices
                cluster_df = features_df[features_df['cluster'] == cluster]
                valid_cluster_indices = [i for i in cluster_df.index if i in valid_indices]
                
                if valid_cluster_indices:
                    # Select random indices from this cluster
                    sample_count = min(samples_per_cluster, len(valid_cluster_indices))
                    selected_indices = np.random.choice(valid_cluster_indices, size=sample_count, replace=False)
                    random_samples.extend(selected_indices)
            
            # Add any remaining samples from random clusters
            remaining_valid = [i for i in valid_indices if i not in random_samples]
            if remaining_samples > 0 and remaining_valid:
                extra_samples = np.random.choice(remaining_valid, 
                                               size=min(remaining_samples, len(remaining_valid)), 
                                               replace=False)
                random_samples.extend(extra_samples)
        else:
            # No clusters, just select random samples from valid indices
            sample_count = min(args.num_samples, len(valid_indices))
            random_samples = np.random.choice(valid_indices, size=sample_count, replace=False)
            
        # If we still don't have enough samples, try to add more from any valid indices
        if len(random_samples) < args.num_samples and len(valid_indices) > len(random_samples):
            additional_indices = [i for i in valid_indices if i not in random_samples]
            additional_count = min(args.num_samples - len(random_samples), len(additional_indices))
            if additional_count > 0:
                additional_samples = np.random.choice(additional_indices, size=additional_count, replace=False)
                random_samples = np.concatenate([random_samples, additional_samples])
        
        print(f"Selected {len(random_samples)} samples for GIF creation")
        
        # Create GIFs for selected samples
        print(f"Creating GIFs for {len(random_samples)} samples...")
        gif_paths = {}
        
        # Dictionary to store max slices for each sample
        max_slices_data = {}
        
        for idx in random_samples:
            try:
                # Validate index is within dataset range before accessing
                if hasattr(dataset, '__len__') and idx >= len(dataset):
                    print(f"Skipping sample {idx} as it is out of bounds for dataset with length {len(dataset)}")
                    continue
                    
                # Get the sample from the dataset
                sample_data = dataset[idx]
                
                # Extract volume data (assuming dataset returns a tuple or has a standard format)
                if isinstance(sample_data, tuple) and len(sample_data) > 0:
                    volume = sample_data[0]  # First element is typically the volume
                    if len(sample_data) > 2:
                        bbox_name = sample_data[2]  # Third element might be bbox_name
                    else:
                        bbox_name = None
                elif isinstance(sample_data, dict):
                    volume = sample_data.get('pixel_values', sample_data.get('raw_volume'))
                    bbox_name = sample_data.get('bbox_name')
                else:
                    volume = sample_data
                    bbox_name = None
                
                # If bbox_name not found in sample_data, try to get it from features_df
                if not bbox_name and idx in features_df.index:
                    bbox_name = features_df.loc[idx].get('bbox_name', 'unknown')
                
                # Skip if no volume data found or it's None/empty
                if volume is None or (hasattr(volume, 'numel') and volume.numel() == 0) or \
                   (hasattr(volume, 'size') and np.prod(volume.shape) == 0):
                    print(f"Skipping sample {idx}: No valid volume data")
                    continue
                
                # Calculate max slices if we have bbox_name
                if bbox_name:
                    print(f"Calculating max slices for sample {idx}, bbox {bbox_name}")
                    max_slices = extract_max_slices_simple(volume)
                    if max_slices:
                        max_slices_data[idx] = max_slices
                        print(f"Calculated max slices for sample {idx}, bbox {bbox_name}: {max_slices}")
                    else:
                        print(f"Warning: Failed to calculate max slices for sample {idx}, bbox {bbox_name}")
                else:
                    print(f"Warning: No bbox_name available for sample {idx}, cannot calculate max slices")
                
                # Create GIF
                sample_info = features_df.loc[idx]
                bbox_name = sample_info.get('bbox_name', 'unknown')
                var1 = sample_info.get('Var1', f'sample_{idx}')
                
                # Clean any problematic characters from filename
                clean_var1 = str(var1).replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
                
                gif_filename = f"{bbox_name}_{clean_var1}_{idx}.gif"
                gif_path = gifs_dir / gif_filename
                
                # Generate GIF with reduced quality to save space
                gif_path, frames = create_gif_from_volume(volume, str(gif_path), fps=8, segmentation_type=None)
                
                # Check if GIF was successfully created
                if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                    # Store full absolute path for HTML file - this is crucial for browser to find the GIFs
                    gif_paths[idx] = os.path.abspath(str(gif_path))
                    # Store frame data for the global slider
                    if 'all_frames' not in locals():
                        all_frames = {}
                    all_frames[idx] = frames
                    print(f"Created GIF for sample {idx} with {len(frames)} frames")
                else:
                    print(f"Failed to create GIF for sample {idx} - file not created or empty")
                
            except Exception as e:
                print(f"Error creating GIF for sample {idx}: {str(e)}")
        
        # Now create visualizations with our simpler methods if we have GIFs
        if gif_paths:                    
            # print("\nCreating animated GIF visualization...")
            # print(f"GIF paths: {list(gif_paths.keys())}")
            # print(f"Features df columns: {features_df.columns.tolist()}")
            # print(f"Features df index range: {features_df.index.min()} to {features_df.index.max()}")
            # print(f"Output directory: {output_dir}")
            
            try:
                # Pass max_slices_data to the visualization function
                animated_path = create_animated_gif_visualization(
                    features_df, gif_paths, output_dir, 
                    dim_reduction=args.dim_reduction, 
                    frame_data=all_frames,
                    max_slices_data=max_slices_data
                )
                print(f"Animated GIF visualization created at {animated_path}")
                print(f"Open this in your browser to see animated GIFs directly at their {args.dim_reduction.upper()} coordinates.")
            except Exception as e:
                print(f"Error creating animated GIF visualization: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No GIFs were created successfully. Skipping additional visualizations.")
    else:
        print("No valid indices found. Skipping GIF creation and visualizations.")

if __name__ == "__main__":
    main()