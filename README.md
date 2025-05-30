# SimCLR for Synaptic EM Data

This project implements SimCLR self-supervised learning for 3D electron microscopy (EM) data of synapses, following the approach described in [Chen et al., 2020](https://arxiv.org/abs/2002.05709) with adaptations for 3D volumetric data.

## Project Overview

This pipeline processes labeled 3D EM data of synapses, with the following workflow:

1. **Preprocessing**: Standardize multiple input sources (raw EM, segmentation, additional masks) into 3-channel volumes.
2. **Dynamic Sampling**: Extract random 80³ cubes from the volumes.
3. **Augmentation**: Apply two different augmentation streams to create positive pairs.
4. **Contrastive Learning**: Use a SimCLR approach with NT-Xent loss.
5. **Evaluation**: Assess embedding quality using silhouette score and UMAP visualization.

## Input Data Structure

The model processes 3D electron microscopy data of synapses with the following characteristics:

### Raw Input Files
- **Raw EM Data**: Grayscale TIFF slices showing cellular ultrastructure
- **Segmentation Masks**: Instance segmentation masks as TIFF slices
- **Additional Masks**: Semantic segmentation masks for specific cellular structures
- **Coordinate Files**: Excel files with coordinates for synaptic cleft centers and sides

### Preprocessed 3-Channel Format
Each volume is converted to a standardized 3-channel tensor:
1. **Channel 0**: Raw EM intensity (normalized)
2. **Channel 1**: Pre-synaptic mask (derived from segmentation)
3. **Channel 2**: Cleft mask (from additional masks)

### Training Inputs
During training, the model receives:
- **Positive Pairs**: Two different augmentations of the same 80³ cube
  - **View A**: Global transformations (rotations, intensity jitter, blur)
  - **View B**: Global + local transformations (crop-resize, mask dropout)
- **Mask-Aware Sampling**: "SegCLR trick" ensures positive pairs share the same biological structures by:
  - Sampling cubes that overlap with the same cleft components
  - Verifying both augmented views preserve sufficient cleft structure

Each bbox volume has shape `[3, 575, 575, 575]`, containing 7 different synapses that are processed into the standardized format.

## Main Features

- Process raw EM data, segmentation masks, and additional mask data into structured 3-channel tensors
- Support for mask-aware contrastive learning that leverages biological structures
- Efficient implementation with mixed precision, gradient accumulation, and distributed training
- Comprehensive evaluation tools for assessing embedding quality without labeled data
- Flexible architecture with ResNet3D or Swin-Transformer3D backbone options

## Directory Structure

```
.
├── datasets/
│   ├── bbox_loader.py         # Load and preprocess bbox volumes
│   └── random_cube.py         # Random cube sampling and augmentation
├── models/
│   └── simclr3d.py            # SimCLR model architecture and loss
├── data/                      # Input data directory
│   ├── raw/                   # Raw EM data
│   │   └── bbox{1-7}/         # Raw TIFF slices for each bbox
│   ├── seg/                   # Segmentation masks
│   │   └── bbox{1-7}/         # Segmentation TIFF slices
│   ├── bbox_{1-7}/            # Additional mask data
│   │   └── slice_*.tif        # Masks for vesicles, clefts, mitochondria
│   └── bbox{1-7}.xlsx         # Coordinate information for each bbox
├── preproc/                   # Preprocessed data (H5 files)
├── checkpoints/               # Model checkpoints
├── eval_results/              # Evaluation results
├── train_simclr.py            # Main training script
├── eval_embeddings.py         # Embedding evaluation script
└── requirements.txt           # Dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd simclr-synapse
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

The code expects the following data structure:
- `data/raw/bbox{1-7}/` - Raw EM data TIFF slices
- `data/seg/bbox{1-7}/` - Segmentation mask TIFF slices
- `data/bbox_{1-7}/` - Additional mask TIFF slices (vesicle, cleft, mitochondria)
- `data/bbox{1-7}.xlsx` - Excel files with coordinate information

### Preprocessing

Preprocessing happens automatically during training or evaluation, but you can run it separately:

```bash
python -c "from datasets.bbox_loader import BBoxLoader; BBoxLoader().process_all_bboxes()"
```

### Training

Basic training:
```bash
python train_simclr.py --data_dir data --preproc_dir preproc --batch_size 64 --epochs 200
```

Advanced training with mask-aware sampling:
```bash
python train_simclr.py --data_dir data --preproc_dir preproc \
  --backbone resnet3d --batch_size 32 --epochs 200 \
  --gradient_accumulation_steps 4 --mixed_precision \
  --mask_aware_after_epoch 100
```

Multi-GPU training:
```bash
python train_simclr.py --distributed --gpus_per_node 4 --batch_size 32 \
  --gradient_accumulation_steps 2 --mixed_precision
```

With Weights & Biases tracking:
```bash
python train_simclr.py --use_wandb --wandb_project simclr-3d-synapse \
  --batch_size 32 --mixed_precision --mask_aware
```

### Evaluation

Evaluate embeddings quality:
```bash
python eval_embeddings.py --checkpoint checkpoints/checkpoint-best.pt \
  --data_dir data --preproc_dir preproc --output_dir eval_results
```

## Arguments

### Training Arguments

- `--data_dir`: Directory with raw data (default: "data")
- `--preproc_dir`: Directory for preprocessed data (default: "preproc")
- `--batch_size`: Batch size per GPU (default: 64)
- `--epochs`: Number of training epochs (default: 200)
- `--backbone`: Backbone model (choices: "resnet3d", "swin3d", default: "resnet3d")
- `--mixed_precision`: Use mixed precision training
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients
- `--mask_aware`: Use mask-aware positive sampling
- `--mask_aware_after_epoch`: Switch to mask-aware sampling after this epoch
- `--distributed`: Use distributed training
- `--use_wandb`: Use Weights & Biases for logging
- `--wandb_project`: Wandb project name (default: "simclr-3d-synapse")
- `--wandb_run_name`: Wandb run name (default: auto-generated)

### Evaluation Arguments

- `--checkpoint`: Path to model checkpoint
- `--max_samples`: Maximum number of samples to evaluate
- `--n_clusters`: Number of clusters for K-means

## Expected Results

After 150-200 epochs of training, we should expect:
- Silhouette score > 0.15
- Visible clusters in UMAP visualization
- Hopkins statistic > 0.6

## References

- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020.
- Lu, Y., Koohababni, N.A., et al. (2022). Self-supervised learning with masked image modeling for instance segmentation and classification in electron microscopy. Nature Machine Intelligence. # Synapseclr
