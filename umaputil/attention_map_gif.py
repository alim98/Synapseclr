# Add project root to sys.path BEFORE other imports
import sys
from pathlib import Path

# Resolve project root and add to path so local packages can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from typing import Any, Dict

from models.simclr3d import SimCLR
from umaputil.GifUmap import initialize_synapse_dataset


def compute_saliency_map(model: SimCLR, volume: torch.Tensor) -> torch.Tensor:
    """Compute absolute gradient saliency map for a single volume.

    Args:
        model: Trained SimCLR model.
        volume: Tensor of shape (C, D, H, W) in range [0, 1] or [0, 255].

    Returns:
        saliency: Tensor of shape (D, H, W) in range [0, 1].
    """
    model.eval()
    volume = volume.unsqueeze(0).requires_grad_(True)  # (1, C, D, H, W)

    # Forward pass through backbone **with gradients enabled** (don't use get_features which has no_grad)
    embedding = model.backbone(volume)  # (1, feat_dim)
    scalar = embedding.pow(2).sum()  # use L2 norm of embedding as target
    scalar.backward()

    # gradient w.r.t input
    grad = volume.grad.detach().abs()  # (1, C, D, H, W)
    grad = grad.sum(dim=1).squeeze(0)  # (D, H, W)
    # normalize per volume
    grad = grad / (grad.max() + 1e-8)
    return grad


def overlay_heatmap(raw_slice: np.ndarray, heat_slice: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay heatmap on grayscale slice.

    Args:
        raw_slice: 2-D uint8 array (H, W).
        heat_slice: 2-D float array in [0, 1] (H, W).

    Returns:
        3-D uint8 RGB image.
    """
    base = np.stack([raw_slice] * 3, axis=2).astype(np.float32)
    cmap = plt.get_cmap('jet')
    heat_color = (cmap(heat_slice)[:, :, :3] * 255).astype(np.float32)
    blended = np.clip((1 - alpha) * base + alpha * heat_color, 0, 255).astype(np.uint8)
    return blended


def create_attention_gif(volume: torch.Tensor, saliency: torch.Tensor, output_path: Path, fps: int = 10):
    """Create gif with attention overlay.

    Args:
        volume: (C, D, H, W) tensor.
        saliency: (D, H, W) tensor in [0, 1].
        output_path: Path to save gif.
    """
    # ensure raw channel 0 is uint8
    vol_np = volume[0].cpu().numpy()
    if vol_np.max() <= 1.0:
        vol_np = (vol_np * 255.0).astype(np.uint8)
    else:
        vol_np = np.clip(vol_np, 0, 255).astype(np.uint8)

    sal_np = saliency.cpu().numpy()

    frames = []
    for z in range(vol_np.shape[0]):
        raw_slice = vol_np[z]
        heat_slice = sal_np[z]
        frame = overlay_heatmap(raw_slice, heat_slice)
        frames.append(frame)

    imageio.mimsave(output_path, frames, fps=fps)


def safe_torch_load(path: str, map_location: str = 'cpu') -> Dict[str, Any]:
    """Robust wrapper around torch.load that copes with the new weights_only=True default in PyTorch 2.6.

    It first tries the default behaviour; if that fails with an UnpicklingError or RuntimeError
    mentioning `weights_only`, it retries with `weights_only=False` – which restores the old
    (pre-2.6) behaviour that allows arbitrary pickled objects. **Only do this if you trust the
    checkpoint source**, which we do for our locally trained model.
    """
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e1:
        print(f"Primary torch.load failed ({type(e1).__name__}: {e1}). Retrying with weights_only=False …")
        try:
            return torch.load(path, map_location=map_location, weights_only=False)  # PyTorch ≥2.6
        except TypeError:
            # For older PyTorch versions that don't recognise weights_only, re-raise original
            raise e1


# -----------------------------------------------------------------------------
# Utility: extract the model state_dict from a checkpoint of various formats
# -----------------------------------------------------------------------------


def get_state_dict(ckpt: dict):
    """Return the most plausible state_dict from an arbitrary checkpoint dict."""
    # Standard lightning checkpoints use key 'state_dict'
    if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        return ckpt['state_dict']
    # Common torch.save wrappers
    if 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
        return ckpt['model_state_dict']
    # Maybe the checkpoint itself *is* the state dict
    if all(isinstance(k, str) and (k.startswith('backbone') or k.startswith('projector') or 'weight' in k) for k in ckpt.keys()):
        return ckpt
    raise RuntimeError("Could not locate a valid state_dict in checkpoint. Keys: " + ", ".join(ckpt.keys()))


def filter_compatible_keys(state: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return subset of state dict whose keys exist in the model **and** match tensor shape."""
    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            if k not in model_state:
                print(f"Skipping '{k}' (not present in model)")
            else:
                print(f"Skipping '{k}' (shape mismatch {v.shape} vs {model_state[k].shape})")
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Generate attention map GIFs for synapse samples.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained SimCLR checkpoint (.pt)')
    parser.add_argument('--preproc_dir', type=str, default='preproc_synapses', help='Directory with bbox*_synapses.h5 files')
    parser.add_argument('--output_dir', type=str, default='attention_gifs', help='Directory to save output GIFs')
    parser.add_argument('--cube_size', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (robustly handle PyTorch ≥2.6 weights_only behaviour)
    checkpoint = safe_torch_load(args.checkpoint, map_location=args.device)
    state = get_state_dict(checkpoint)

    model = SimCLR(in_channels=3, pretrained=False)
    compat_state = filter_compatible_keys(state, model)
    missing, unexpected = model.load_state_dict(compat_state, strict=False)
    if missing:
        print(f"WARNING: Missing {len(missing)} keys when loading state_dict, model may be partially initialised.")
    if unexpected:
        print(f"WARNING: Unexpected {len(unexpected)} keys ignored when loading state_dict.")
    model.to(args.device)

    # Load dataset
    dataset, _ = initialize_synapse_dataset(preproc_dir=args.preproc_dir, cube_size=args.cube_size)
    if dataset is None:
        print("Failed to load dataset.")
        return

    # Map from bbox number to first sample id
    samples_by_bbox = {}
    for syn_id in dataset.synapse_ids:
        if syn_id.startswith('bbox'):
            bbox_name = syn_id.split('_')[0]  # e.g., bbox3
            if bbox_name not in samples_by_bbox:
                samples_by_bbox[bbox_name] = syn_id
        if len(samples_by_bbox) == 7:
            break

    if not samples_by_bbox:
        print('No samples found.')
        return

    for bbox_name, syn_id in samples_by_bbox.items():
        cube = dataset.synapse_cubes[syn_id].to(args.device)
        saliency = compute_saliency_map(model, cube)
        gif_path = output_dir / f"{bbox_name}_{syn_id}_attention.gif"
        print(f"Creating attention GIF for {syn_id} -> {gif_path}")
        create_attention_gif(cube.cpu(), saliency.cpu(), gif_path)


if __name__ == '__main__':
    main() 