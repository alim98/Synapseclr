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
from datasets.bbox_loader import BBoxLoader
from datasets.random_cube import RandomCubeDataset


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


def extract_embeddings(model: torch.nn.Module, dataset: RandomCubeDataset, 
                       device: torch.device, max_samples: int = 100000,
                       batch_size: int = 32, mask_overlap_threshold: float = 0.4) -> np.ndarray:
    """
    Extract embeddings from the model for a subset of dataset samples.
    
    Args:
        model: SimCLR model (only backbone will be used)
        dataset: RandomCubeDataset for sampling cubes
        device: Device to run the model on
        max_samples: Maximum number of samples to extract embeddings for
        batch_size: Batch size for inference
        mask_overlap_threshold: Minimum vesicle mask overlap required
        
    Returns:
        Numpy array of embeddings with shape (n_samples, embedding_dim)
    """
    model.eval()
    embeddings = []
    
    # Determine how many samples per bbox
    num_bboxes = len(dataset.bbox_names)
    samples_per_bbox = max(1, max_samples // num_bboxes)
    
    print(f"Extracting embeddings from {num_bboxes} bboxes, "
          f"{samples_per_bbox} samples per bbox...")
    
    # Process each bbox to balance the dataset
    with torch.no_grad():
        for bbox_name in tqdm(dataset.bbox_names):
            # Count samples for this bbox
            bbox_samples = 0
            
            # Create a batch
            batch = []
            
            # Sample cubes from this bbox
            while bbox_samples < samples_per_bbox:
                # Sample from this specific bbox
                try:
                    cube, _ = dataset._sample_valid_cube(bbox_name)
                    
                    # Check if the cube has significant vesicle content
                    vesicle_mask = cube[dataset.vesicle_channel]
                    if torch.sum(vesicle_mask) / torch.numel(vesicle_mask) < mask_overlap_threshold / 10:
                        continue  # Skip cubes with minimal vesicle content
                    
                    # Add to batch
                    batch.append(cube)
                    bbox_samples += 1
                    
                    # Process batch if full
                    if len(batch) == batch_size or bbox_samples == samples_per_bbox:
                        # Stack and move to device
                        batch_tensor = torch.stack(batch).to(device)
                        
                        # Extract features
                        features = model.get_features(batch_tensor)
                        
                        # Move to CPU and convert to numpy
                        features = features.cpu().numpy()
                        embeddings.append(features)
                        
                        # Reset batch
                        batch = []
                except Exception as e:
                    print(f"Error sampling from {bbox_name}: {e}")
                    break
    
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
                        choices=['resnet3d', 'swin3d'],
                        help='Backbone model (default: resnet3d)')
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
    model = SimCLR(backbone_type=args.backbone)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle both full model and backbone-only checkpoints
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and not 'epoch' in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        # Assume it's just the backbone
        model.backbone.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Load data
    print("Loading bbox data...")
    loader = BBoxLoader(
        data_dir=args.data_dir,
        preproc_dir=args.preproc_dir,
        create_h5=True
    )
    volumes = loader.process_all_bboxes()
    
    # Create dataset for sampling random cubes
    dataset = RandomCubeDataset(
        bbox_volumes=volumes,
        cubes_per_bbox=args.max_samples // len(volumes) + 1,
        cube_size=args.cube_size,
        mask_aware=False  # Don't need mask-aware for evaluation
    )
    
    # Extract embeddings
    embeddings = extract_embeddings(
        model,
        dataset,
        device,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
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