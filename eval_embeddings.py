import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import h5py
import pandas as pd
from tqdm import tqdm

# Local imports
from models.simclr3d import SimCLR
from datasets.synapse_loader import SynapseLoader
from datasets.synapse_dataset import SynapseDataset


def hopkins_statistic(X: np.ndarray, m: int = None) -> float:
    """
    Compute the Hopkins statistic for clustering tendency.
    
    Parameters:
    -----------
    X : array_like
        Data matrix with shape (n_samples, n_features)
    m : int, optional
        Number of sample points to use (default: 0.1 * n_samples)
        
    Returns:
    --------
    h : float
        Hopkins statistic (between 0 and 1)
        Values close to 1 indicate data is highly clusterable
        Values around 0.5 indicate data is random
        Values close to 0 indicate uniform data
    """
    n, d = X.shape
    m = m or int(0.1 * n)
    
    # Generate random points from uniform distribution over range of X
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
    # Generate uniform random points
    uniform_points = np.random.uniform(mins, maxs, size=(m, d))
    
    # Randomly select m sample points from X
    sample_idx = np.random.choice(n, m, replace=False)
    sample_points = X[sample_idx]
    
    # Calculate nearest neighbor distances for uniform points
    u_distances = []
    for point in uniform_points:
        # Calculate distances to all points in X
        distances = np.sqrt(np.sum((X - point) ** 2, axis=1))
        u_distances.append(np.min(distances))
    
    # Calculate nearest neighbor distances for sample points
    w_distances = []
    for idx in sample_idx:
        # Calculate distances to all points in X (excluding the point itself)
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        distances = np.sqrt(np.sum((X[mask] - X[idx]) ** 2, axis=1))
        w_distances.append(np.min(distances))
    
    # Convert to arrays
    u_distances = np.array(u_distances)
    w_distances = np.array(w_distances)
    
    # Calculate Hopkins statistic
    h = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))
    
    return h


def extract_embeddings(model: torch.nn.Module, dataset: SynapseDataset, 
                       device: torch.device, max_samples: int = 100000,
                       batch_size: int = 32) -> np.ndarray:
    """
    Extract embeddings from the model for a subset of dataset samples.
    
    Args:
        model: SimCLR model (only backbone will be used)
        dataset: SynapseDataset for synapse cubes
        device: Device to run the model on
        max_samples: Maximum number of samples to extract embeddings for
        batch_size: Batch size for inference
        
    Returns:
        Numpy array of embeddings with shape (n_samples, embedding_dim)
    """
    model.eval()
    embeddings = []
    
    # Limit number of samples if dataset is larger than max_samples
    num_samples = min(len(dataset), max_samples)
    print(f"Extracting embeddings from {num_samples} synapse samples...")
    
    # Create dataloader for efficient batch processing
    from torch.utils.data import DataLoader, Subset
    import random
    
    # Randomly sample indices if we need fewer samples
    if len(dataset) > max_samples:
        indices = random.sample(range(len(dataset)), max_samples)
        subset_dataset = Subset(dataset, indices)
    else:
        subset_dataset = dataset
    
    dataloader = DataLoader(
        subset_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Single-threaded for simplicity
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting embeddings"):
            # Handle both single cube and augmented pairs
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                # If we get augmented pairs, just use the first view
                cubes = batch_data[0]
            else:
                cubes = batch_data
            
            # Move to device
            cubes = cubes.to(device, non_blocking=True)
            
            # Extract features using the backbone
            features = model.get_features(cubes)
            
            # Move to CPU and convert to numpy
            features = features.cpu().numpy()
            embeddings.append(features)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings) if embeddings else np.array([])
    
    print(f"Extracted {all_embeddings.shape[0]} embeddings with dimension {all_embeddings.shape[1]}")
    return all_embeddings


def analyze_embeddings(embeddings: np.ndarray, n_clusters: int = 100, 
                       random_state: int = 42) -> Tuple[float, float]:
    """
    Analyze embeddings using clustering metrics.
    
    Args:
        embeddings: Embedding array with shape (n_samples, embedding_dim)
        n_clusters: Number of clusters for K-means
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (silhouette_score, hopkins_statistic)
    """
    print(f"Analyzing {embeddings.shape[0]} embeddings...")
    
    # Skip if too few samples
    if embeddings.shape[0] < n_clusters + 1:
        print(f"Too few samples ({embeddings.shape[0]}) for {n_clusters} clusters.")
        return 0, 0
    
    # Calculate Hopkins statistic
    h = hopkins_statistic(embeddings)
    print(f"Hopkins statistic: {h:.4f}")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score
    silhouette = silhouette_score(embeddings, cluster_labels)
    print(f"Silhouette score with {n_clusters} clusters: {silhouette:.4f}")
    
    return silhouette, h


def visualize_embeddings(embeddings: np.ndarray, output_dir: str, 
                         n_neighbors: int = 15, min_dist: float = 0.1,
                         random_state: int = 42) -> None:
    """
    Visualize embeddings using UMAP and t-SNE.
    
    Args:
        embeddings: Embedding array with shape (n_samples, embedding_dim)
        output_dir: Directory to save visualizations
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random state for reproducibility
    """
    print("Creating dimensionality reduction visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Subsample if too many embeddings
    max_vis_samples = 10000
    if embeddings.shape[0] > max_vis_samples:
        idx = np.random.choice(embeddings.shape[0], max_vis_samples, replace=False)
        vis_embeddings = embeddings[idx]
    else:
        vis_embeddings = embeddings
    
    # UMAP
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='euclidean',
        random_state=random_state
    )
    umap_embeddings = umap_reducer.fit_transform(vis_embeddings)
    
    # t-SNE
    tsne_embeddings = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=random_state
    ).fit_transform(vis_embeddings)
    
    # Plot UMAP
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5, alpha=0.5)
    plt.title("UMAP Visualization of Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_embeddings.png"), dpi=300)
    plt.close()
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=5, alpha=0.5)
    plt.title("t-SNE Visualization of Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_embeddings.png"), dpi=300)
    plt.close()
    
    # K-means clustering on UMAP embeddings
    kmeans = KMeans(n_clusters=10, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(umap_embeddings)
    
    # Plot UMAP with clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                         c=clusters, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title("UMAP with K-means Clusters (k=10)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_clusters.png"), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SimCLR embeddings')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing raw data')
    parser.add_argument('--preproc_dir', type=str, default='preproc',
                        help='Directory with preprocessed data')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet3d',
                        help='Backbone model (only resnet3d supported)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension used in the training projector (default: 512)')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Output projection dimension (default: 128)')
    parser.add_argument('--cube_size', type=int, default=80,
                        help='Size of cube to sample (default: 80)')
    
    # Evaluation parameters
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--n_clusters', type=int, default=100,
                        help='Number of clusters for K-means')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    # Instantiate with user-specified projector dims so it matches training
    model = SimCLR(in_channels=3, hidden_dim=args.hidden_dim, out_dim=args.output_dim)
    
    # ------------------------------------------------------------------
    # PyTorch >=2.6 changed the default of `weights_only` to True which can
    # break loading checkpoints that contain pickled Python objects such as
    # `argparse.Namespace`.  We first attempt the safe default (weights_only
    # = True) and transparently fall back to the old behaviour if that fails.
    # ------------------------------------------------------------------
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except Exception as e:
        # Typical message: "Weights only load failed"  – retry with
        # weights_only=False **only if you trust the checkpoint's source**.
        print(f"Warning: safe checkpoint load failed ({e}). Retrying with weights_only=False...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Robust checkpoint loading
    # ------------------------------------------------------------------
    def _load_backbone_from(state_dict: dict):
        backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
        missing = model.backbone.load_state_dict(backbone_state, strict=False)
        if missing.missing_keys:
            print(f"Warning: Missing backbone keys when loading checkpoint: {missing.missing_keys}")

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Full model load failed due to size mismatch ({e}). Loading backbone only...")
            _load_backbone_from(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and not 'epoch' in checkpoint:
        # Might be full or backbone-only state_dict
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"State-dict load failed ({e}). Attempting backbone-only load...")
            _load_backbone_from(checkpoint)
    else:
        # Assume it's just the backbone weights as raw state_dict
        try:
            model.backbone.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"Backbone load still failed ({e}). Please check the checkpoint file.")
    
    model = model.to(device)
    
    # Load synapse data (same as training)
    print("Loading synapse data...")
    loader = SynapseLoader(
        data_dir=args.data_dir,
        preproc_dir=args.preproc_dir,
        cube_size=args.cube_size
    )
    
    # Process all synapses (use memory efficient loading if available)
    try:
        synapses = loader.process_all_synapses(memory_efficient=True)
    except Exception as e:
        print(f"Failed to load synapses from Excel files: {e}")
        print("Falling back to loading directly from H5 files...")
        synapses = {}
    
    # If memory_efficient was used, load from paths
    if synapses and isinstance(next(iter(synapses.values())), str):
        print("Loading synapses from H5 files...")
        synapses = loader.load_from_paths(synapses)
        
        # Verify all synapses loaded successfully
        failed_loads = [name for name, syn in synapses.items() if syn is None]
        if failed_loads:
            print(f"Warning: Failed to load some synapses: {failed_loads}")
            # Remove failed loads
            synapses = {name: syn for name, syn in synapses.items() if syn is not None}
    
    # Fallback: Load directly from H5 files if no synapses found
    if len(synapses) == 0:
        print("No synapses loaded from Excel approach. Loading directly from H5 files...")
        import h5py
        for i in range(1, 8):
            h5_file = os.path.join(args.preproc_dir, f"bbox{i}_synapses.h5")
            if os.path.exists(h5_file):
                try:
                    with h5py.File(h5_file, 'r') as f:
                        for key in f.keys():
                            # Read the synapse cube
                            cube_data = f[key][:]
                            cube_tensor = torch.from_numpy(cube_data).float()
                            # Use bbox prefix for consistency with training
                            synapse_id = f"bbox{i}_{key}"
                            synapses[synapse_id] = cube_tensor
                    print(f"Loaded {len(list(h5py.File(h5_file, 'r').keys()))} synapses from bbox{i}")
                except Exception as e:
                    print(f"Error loading H5 file {h5_file}: {e}")
    
    print(f"Loaded {len(synapses)} synapse cubes")
    
    if len(synapses) == 0:
        print("ERROR: No synapses found! Cannot proceed with evaluation.")
        return
    
    # Create dataset (no augmentation for evaluation)
    dataset = SynapseDataset(
        synapse_cubes=synapses,
        augment=False  # No augmentation for evaluation
    )
    
    # Extract embeddings
    embeddings = extract_embeddings(
        model,
        dataset,
        device,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Check if we got any embeddings
    if embeddings.size == 0:
        print("ERROR: No embeddings extracted! Cannot proceed with analysis.")
        return
    
    # Save embeddings
    np.save(os.path.join(args.output_dir, 'embeddings.npy'), embeddings)
    
    # Analyze embeddings
    silhouette, hopkins = analyze_embeddings(
        embeddings,
        n_clusters=min(args.n_clusters, len(embeddings) // 2),
        random_state=42
    )
    
    # Visualize embeddings
    visualize_embeddings(
        embeddings,
        output_dir=args.output_dir,
        random_state=42
    )
    
    # Save metrics
    metrics = {
        'silhouette_score': silhouette,
        'hopkins_statistic': hopkins,
        'n_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'checkpoint': args.checkpoint
    }
    
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print("\nEvaluation complete!")
    print(f"Silhouette score: {silhouette:.4f}")
    print(f"Hopkins statistic: {hopkins:.4f}")
    
    # Provide interpretation
    print("\nInterpretation:")
    if silhouette > 0.2:
        print("✓ Strong clustering structure detected (silhouette > 0.2)")
    elif silhouette > 0.1:
        print("✓ Moderate clustering structure detected (silhouette > 0.1)")
    else:
        print("✗ Weak clustering structure (silhouette < 0.1)")
        
    if hopkins > 0.7:
        print("✓ Strong clustering tendency (hopkins > 0.7)")
    elif hopkins > 0.6:
        print("✓ Moderate clustering tendency (hopkins > 0.6)")
    else:
        print("✗ Data may be randomly distributed (hopkins < 0.6)")


if __name__ == '__main__':
    main() 