# SimCLR for Synaptic EM Data
⚡ main ~ python train_synapse_simclr.py --data_dir datareal --epochs 10 --batch_size 8 --cube_size 80 --run_name my_experimentssss

python train_synapse_simclr.py --data_dir datareal --epochs 600 --batch_size 8 --run_name enhanced_training_constant_lr --scheduler_type constant
This project implements SimCLR self-supervised learning for 3D electron microscopy (EM) data of synapses, following the approach described in [Chen et al., 2020](https://arxiv.org/abs/2002.05709) with adaptations for 3D volumetric data.

## Project Overview

This pipeline processes labeled 3D EM data of synapses with two different approaches:

1. **Synapse-Based Training (Recommended)**: Extracts individual synapses based on Excel coordinate annotations
2. **BBox-Based Training**: Processes entire bounding box volumes and randomly samples cubes

### Synapse-Based Approach (New)

This approach extracts specific synapses using coordinates from Excel files:

1. **Coordinate Loading**: Read synapse coordinates (central, side1, side2) from Excel files
2. **Vesicle Analysis**: Determine presynaptic side based on vesicle cloud overlap
3. **Cube Extraction**: Extract 80³ cubes centered on synaptic clefts
4. **Contrastive Learning**: Train SimCLR on individual synapse instances

### BBox-Based Approach (Original)

The original approach processes entire bounding box volumes:

1. **Preprocessing**: Standardize multiple input sources into 3-channel volumes
2. **Dynamic Sampling**: Extract random 80³ cubes from volumes
3. **Augmentation**: Apply augmentation streams to create positive pairs
4. **Contrastive Learning**: Use SimCLR with NT-Xent loss

## Input Data Structure

### Raw Input Files
- **Raw EM Data**: Grayscale TIFF slices showing cellular ultrastructure
- **Segmentation Masks**: Instance segmentation masks as TIFF slices
- **Additional Masks**: Semantic segmentation masks for cellular structures
- **Coordinate Files**: Excel files with synapse coordinates:
  - `Var1`: Synapse identifier
  - `central_coord_1/2/3`: Cleft center coordinates (x, y, z)
  - `side_1_coord_1/2/3`: First synaptic partner coordinates
  - `side_2_coord_1/2/3`: Second synaptic partner coordinates

### Preprocessed 3-Channel Format
Each volume/cube is converted to a standardized 3-channel tensor:
1. **Channel 0**: Raw EM intensity (normalized)
2. **Channel 1**: Pre-synaptic mask (determined by vesicle overlap)
3. **Channel 2**: Cleft mask (from additional masks)

## Directory Structure

```
.
├── datasets/
│   ├── bbox_loader.py         # Load and preprocess bbox volumes
│   ├── synapse_loader.py      # Extract individual synapses from Excel
│   ├── synapse_dataset.py     # Dataset for individual synapses
│   └── random_cube.py         # Random cube sampling and augmentation
├── models/
│   └── simclr.py              # SimCLR model architecture and loss
├── utils/
│   └── training_utils.py      # Training utilities
├── data/                      # Input data directory
│   ├── raw/                   # Raw EM data
│   │   └── bbox{1-7}/         # Raw TIFF slices for each bbox
│   ├── seg/                   # Segmentation masks
│   │   └── bbox{1-7}/         # Segmentation TIFF slices
│   ├── bbox_{1-7}/            # Additional mask data
│   │   └── slice_*.tif        # Masks for vesicles, clefts, mitochondria
│   └── bbox{1-7}.xlsx         # Coordinate information for each bbox
├── preproc/                   # Preprocessed bbox data (H5 files)
├── preproc_synapses/          # Preprocessed synapse cubes (H5 files)
├── checkpoints/               # Model checkpoints
├── tensorboard_logs/          # TensorBoard logs (organized by run)
├── train_simclr.py            # Main training script (bbox-based)
├── train_synapse_simclr.py    # Synapse-based training script
├── test_synapse_extraction.py # Test synapse extraction
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

### Synapse-Based Training (Recommended)

This approach extracts individual synapses based on Excel coordinates:

```bash
# Step 1: Test synapse extraction first
python test_synapse_extraction.py --data_dir data --cube_size 80

# Step 2: Preprocess synapses and save to H5 files (optional but recommended)
python preprocess_synapses.py --data_dir data --preproc_dir preproc_synapses --cube_size 80

# Step 3: Train SimCLR on individual synapses
python train_synapse_simclr.py --data_dir data --epochs 200 --batch_size 32 --cube_size 80

# With mixed precision for faster training
python train_synapse_simclr.py --data_dir data --epochs 200 --batch_size 32 --mixed_precision

# With memory efficient loading (uses H5 files automatically)
python train_synapse_simclr.py --data_dir data --epochs 200 --batch_size 32 --memory_efficient

# With custom run name for TensorBoard
python train_synapse_simclr.py --data_dir data --epochs 200 --batch_size 32 --run_name my_experiment
```

### BBox-Based Training (Original)

Train SimCLR on entire bbox volumes:

```bash
# Basic training
python train_simclr.py --data_dir data --epochs 200 --batch_size 64

# Advanced training with mask-aware sampling
python train_simclr.py --data_dir data --batch_size 32 --epochs 200 \
  --gradient_accumulation_steps 4 --mixed_precision \
  --mask_aware_after_epoch 100

# Multi-GPU training
python train_simclr.py --distributed --batch_size 32 \
  --gradient_accumulation_steps 2 --mixed_precision
```

### TensorBoard Monitoring

View training progress:
```bash
tensorboard --logdir tensorboard_logs
```

Each run creates a separate directory:
- `tensorboard_logs/my_experiment_20231215_143022/`
- `tensorboard_logs/synapse_simclr_20231215_144501/` (default naming)

### Data Preparation

The code expects the following data structure:
- `data/raw/bbox{1-7}/` - Raw EM data TIFF slices
- `data/seg/bbox{1-7}/` - Segmentation mask TIFF slices
- `data/bbox_{1-7}/` - Additional mask TIFF slices (vesicle, cleft, mitochondria)
- `data/bbox{1-7}.xlsx` - Excel files with synapse coordinate information

### Evaluation

Evaluate embeddings quality:
```bash
python eval_embeddings.py --checkpoint checkpoints/checkpoint_best.pt \
  --data_dir data --preproc_dir preproc --output_dir eval_results
```

## Key Arguments

### Synapse-Based Training Arguments

- `--data_dir`: Directory with raw data (default: "data")
- `--preproc_dir`: Directory for preprocessed synapse cubes (default: "preproc_synapses")
- `--cube_size`: Size of synapse cubes to extract (default: 80)
- `--batch_size`: Batch size per GPU (default: 32)
- `--epochs`: Number of training epochs (default: 200)
- `--mixed_precision`: Use mixed precision training
- `--run_name`: Name for this training run

### BBox-Based Training Arguments

- `--data_dir`: Directory with raw data (default: "data")
- `--preproc_dir`: Directory for preprocessed bbox data (default: "preproc")
- `--cubes_per_bbox`: Number of cubes to sample per bbox (default: 10)
- `--cube_size`: Size of cubes to sample (default: 80)
- `--mask_aware`: Use mask-aware positive sampling
- `--mask_aware_after_epoch`: Switch to mask-aware sampling after this epoch

## Model Architecture

- **Backbone**: ResNet3D-50 adapted for 3D EM data
- **Input**: 3-channel 80³ cubes
- **Contrastive Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Temperature**: 0.07 (default)
- **Projector**: MLP with hidden dimension 512, output dimension 128

## Expected Results

After 150-200 epochs of training:
- **Synapse-based**: Better representation of individual synaptic structures
- **BBox-based**: Good general representations but may conflate different synapses
- Silhouette score > 0.15
- Visible clusters in UMAP visualization
- Hopkins statistic > 0.6

## Advantages of Synapse-Based Approach

1. **Biological Relevance**: Learns representations of individual synapses rather than arbitrary cube regions
2. **Precise Localization**: Uses expert-annotated coordinates for accurate synapse centering
3. **Consistent Structure**: Each training sample contains a complete synaptic structure
4. **Better Evaluation**: More meaningful for downstream synaptic analysis tasks

## References

- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020.
- Lu, Y., Koohababni, N.A., et al. (2022). Self-supervised learning with masked image modeling for instance segmentation and classification in electron microscopy. Nature Machine Intelligence.

## Visualizing Preprocessed Samples

Before training, it's recommended to visualize some preprocessed samples to sanity check your data pipeline:

```bash
# Visualize 5 samples with augmentations
python visualize_samples.py \
    --data_root /path/to/your/data \
    --excel_file /path/to/synapse_annotations.xlsx \
    --num_samples 5 \
    --show_augmentations \
    --save_dir ./visualizations \
    --bbox_size 64
```

This will:
- Load synapse cubes using your current preprocessing pipeline
- Display all three channels (raw EM, vesicles, segmentation) 
- Show middle slices in XY, XZ, and YZ planes
- Compare original vs augmented views if `--show_augmentations` is used
- Print detailed statistics about each sample
- Save visualization plots to the specified directory

**Tip**: Check that the vesicle and segmentation channels have reasonable non-zero content and that the augmentations look sensible before starting training.

## Training

Basic training command:
```bash
python train_synapse_simclr.py \
    --data_root /path/to/your/data \
    --excel_file /path/to/synapse_annotations.xlsx \
    --batch_size 32 \
    --epochs 200 \
    --learning_rate 1e-3 \
    --temperature 0.1 \
    --bbox_size 64 \
    --run_name my_experiment
```

## Key Parameters

- `--data_root`: Root directory containing bbox folders
- `--excel_file`: Path to Excel file with synapse annotations
- `--bbox_size`: Size of extracted synapse cubes (default: 64)
- `--batch_size`: Training batch size (default: 32)
- `--temperature`: Temperature parameter for contrastive loss (default: 0.1)
- `--memory_efficient`: Use memory-efficient loading for large datasets
- `--run_name`: Custom name for this training run (for TensorBoard logs)

## Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir tensorboard_logs
```

Training logs are saved with timestamps to separate different runs:
- `checkpoints/tensorboard_logs/{run_name}_{timestamp}/`

## Model Architecture

- **Backbone**: 3D ResNet-18/34/50 (configurable)
- **Input**: 3-channel 3D cubes (raw EM + vesicles + segmentation)  
- **Projection head**: 2-layer MLP with ReLU
- **Output**: 128-dimensional embeddings

## Augmentations

The following augmentations are applied during training:
- Random flips along all axes
- Random 90-degree rotations in XY, XZ, YZ planes
- Gaussian noise (raw channel only)
- Contrast/brightness variations (raw channel only)

## License

[Add your license here]
