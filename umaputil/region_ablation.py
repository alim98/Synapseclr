import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Project imports (ensure ROOT added to sys.path)
import sys
from pathlib import Path as _Path
ROOT_DIR = _Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.simclr3d import SimCLR
from umaputil.GifUmap import initialize_synapse_dataset
from umaputil.attention_map_gif import safe_torch_load, get_state_dict, filter_compatible_keys


################################################################################
# Utility functions
################################################################################

def compute_embedding(model: SimCLR, cube: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return 1-D embedding vector for a single cube using the model backbone."""
    model.eval()
    with torch.no_grad():
        x = cube.unsqueeze(0).to(device)  # (1, C, D, H, W)
        emb = model.backbone(x)
        # Global average pool if spatial dimensions remain
        if emb.ndim > 2:
            emb = F.adaptive_avg_pool3d(emb, (1, 1, 1))
            emb = emb.view(1, -1)
        return emb.squeeze(0).cpu()


def mask_region(cube: torch.Tensor, region: str) -> torch.Tensor:
    """Return a copy of cube with **region** voxels zeroed out on **all channels**.

    Regions:
        presynapse  – voxels where channel-1 mask > 0
        cleft       – voxels where channel-2 mask > 0
        postsynapse – voxels not covered by presynapse or cleft
    """
    assert cube.dim() == 4 and cube.size(0) >= 3, "Cube must be (C, D, H, W) with >=3 channels"
    masked = cube.clone()

    pres_mask = (cube[1] > 0)
    cleft_mask = (cube[2] > 0)

    if region == 'presynapse':
        region_mask = pres_mask
    elif region == 'cleft':
        region_mask = cleft_mask
    elif region == 'postsynapse':
        region_mask = (~pres_mask) & (~cleft_mask)
    else:
        raise ValueError(f"Unknown region: {region}")

    # Zero out region on all channels
    for c in range(masked.size(0)):
        masked[c][region_mask] = 0.0

    return masked


def embedding_difference(emb1: torch.Tensor, emb2: torch.Tensor, metric: str = 'l2') -> float:
    """Return scalar distance between two embeddings."""
    if metric == 'l2':
        return torch.norm(emb1 - emb2).item()
    elif metric == 'cos':
        return 1 - torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    else:
        raise ValueError(f"Unsupported metric: {metric}")

################################################################################
# Main evaluation routine
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Region ablation study: measure how masking synapse regions affects embeddings.")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_latest.pt', help='Path to trained SimCLR checkpoint')
    parser.add_argument('--preproc_dir', type=str, default='preproc_synapses', help='Directory with preprocessed H5 files')
    parser.add_argument('--cube_size', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--metric', type=str, choices=['l2', 'cos'], default='l2', help='Distance metric to quantify embedding change')
    parser.add_argument('--outfile', type=str, default='region_ablation_results.csv', help='CSV to save results')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---------------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------------
    ckpt = safe_torch_load(args.checkpoint, map_location=device)
    state = get_state_dict(ckpt)

    model = SimCLR(in_channels=3, pretrained=False)
    compat = filter_compatible_keys(state, model)
    model.load_state_dict(compat, strict=False)
    model.to(device)

    # ---------------------------------------------------------------------
    # Load dataset and pick one sample per bbox
    # ---------------------------------------------------------------------
    dataset, _ = initialize_synapse_dataset(preproc_dir=args.preproc_dir, cube_size=args.cube_size)
    if dataset is None:
        print("Failed to load dataset – aborting.")
        return

    samples_by_bbox: Dict[str, str] = {}
    for syn_id in dataset.synapse_ids:
        if syn_id.startswith('bbox'):
            bbox = syn_id.split('_')[0]
            if bbox not in samples_by_bbox:
                samples_by_bbox[bbox] = syn_id
        if len(samples_by_bbox) == 7:
            break

    if not samples_by_bbox:
        print("No bbox samples found.")
        return

    print(f"Selected {len(samples_by_bbox)} samples (one per bbox) for ablation study")

    # ---------------------------------------------------------------------
    # Evaluate effect of masking regions
    # ---------------------------------------------------------------------
    results = []  # List[Dict[str, Any]]

    regions = ['postsynapse', 'presynapse', 'cleft']

    for bbox, syn_id in samples_by_bbox.items():
        cube = dataset.synapse_cubes[syn_id].to(device)

        orig_emb = compute_embedding(model, cube, device)

        for region in regions:
            masked_cube = mask_region(cube, region)
            emb_masked = compute_embedding(model, masked_cube, device)
            diff = embedding_difference(orig_emb, emb_masked, metric=args.metric)

            results.append({
                'bbox': bbox,
                'synapse_id': syn_id,
                'region': region,
                'distance': diff
            })

            print(f"[bbox {bbox}] {syn_id}: Masked {region:<11} -> {args.metric} distance = {diff:.4f}")

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(args.outfile, index=False)
    print(f"Saved results to {args.outfile}")

    # Summarise per-region effect
    summary = df.groupby('region')['distance'].mean().sort_values(ascending=False)
    print("\nAverage embedding change per region:")
    print(summary)


if __name__ == '__main__':
    main() 