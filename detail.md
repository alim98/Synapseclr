# SimCLR for 3D Synapse Data: Detailed Implementation

This document provides a comprehensive explanation of the entire codebase for self-supervised representation learning on 3D electron microscopy (EM) data of synapses using SimCLR.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [File Structure and Implementation Details](#file-structure-and-implementation-details)
4. [Training Process](#training-process)
5. [Advanced Features](#advanced-features)
6. [Logging and Visualization](#logging-and-visualization)
7. [Technical Challenges and Solutions](#technical-challenges-and-solutions)

## Project Overview

This project implements a self-supervised learning approach for 3D electron microscopy (EM) data of synapses using the SimCLR (Simple Framework for Contrastive Learning of Visual Representations) methodology. The goal is to learn meaningful representations of synapse structures without requiring manual labels, which can later be used for downstream tasks such as segmentation or classification.

The implementation includes:
- Custom data loading and preprocessing for EM data with structure masks
- 3D-adapted SimCLR with specialized augmentations for volumetric data
- Training with mixed precision, gradient accumulation, and distributed capabilities
- Structure-aware contrastive learning utilizing cleft masks
- Integration with Weights & Biases for experiment tracking

## Data Processing Pipeline

### Input Data Structure

The pipeline starts with three types of raw data:
1. **Raw EM Data**: Grayscale TIFF slices showing cellular ultrastructure
2. **Segmentation Masks**: Instance segmentation TIFF slices identifying different cellular components
3. **Additional Masks**: Semantic segmentation TIFF slices for specific structures (cleft, mitochondria, vesicles)
4. **Coordinate Files**: Excel files with coordinates for synaptic cleft centers and pre/post sides

### Preprocessing Steps (BBoxLoader)

The `BBoxLoader` class in `bbox_loader.py` performs the following steps:
1. **Loading**: Reads TIFF slices and Excel coordinates for each bbox
2. **Standardization**: Handles multi-channel images and converts to grayscale if needed
3. **Normalization**: Z-score normalization of raw EM data
4. **Mask Creation**: 
   - Extracts cleft mask from additional masks
   - Identifies pre-synaptic membrane from segmentation using coordinates
5. **Channel Composition**: Creates a standardized 3-channel tensor:
   - Channel 0: Normalized raw EM data
   - Channel 1: Pre-synaptic mask
   - Channel 2: Cleft mask
6. **Caching**: Saves processed volumes to H5 files for faster loading

### Dynamic Sampling (RandomCubeDataset)

The `RandomCubeDataset` class in `random_cube.py` handles:
1. **Random Sampling**: Extracts 80³ voxel cubes from larger volumes
2. **Mask-Aware Sampling**: Optional feature to ensure cleft structures are present in samples
3. **Component Analysis**: When mask-aware is enabled:
   - Identifies connected components in cleft masks
   - Ensures sampled regions contain significant cleft structures
   - Verifies that augmentations preserve these structures
4. **Positive Pair Generation**: Creates two differently augmented views of the same cube

### Augmentation (Augmenter)

The `Augmenter` class in `random_cube.py` implements specialized 3D augmentations:
1. **Stream A** (applied to both views):
   - 3D rotations (90° increments around random axes)
   - Intensity jitter (only applied to raw channel)
   - 3D Gaussian blur (slice-by-slice 2D implementation)
2. **Stream B** (additional transformations for second view):
   - Random crop-resize (varies cube size then resizes back)
   - Mask dropout (occasionally zeros out the cleft mask)

## File Structure and Implementation Details

### datasets/bbox_loader.py

This file handles loading and preprocessing of the raw data:

1. **BBoxLoader Class**:
   - **Constructor**: Initializes paths and settings
   - **load_excel_data**: Loads coordinate information from Excel files
   - **load_volumes**: Loads the three volumes (raw, segmentation, additional masks)
   - **get_closest_component_mask**: Finds connected components closest to a point
   - **create_segment_masks**: Creates binary masks for pre/post sides
   - **process_bbox**: Main processing function that creates the 3-channel tensor
   - **process_all_bboxes**: Processes all bbox volumes in the dataset
   
2. **Key Implementation Details**:
   - **LABEL_MAP**: Maps bbox numbers to label values in the annotation files
   - **H5 Caching**: Stores processed volumes in H5 format for performance
   - **Error Handling**: Robust handling of missing files or corrupt data

### datasets/random_cube.py

This file handles dynamic sampling and augmentation:

1. **Augmenter Class**:
   - **Constructor**: Sets up augmentation parameters
   - **apply_rotation**: Implements 3D rotation with 90° increments
   - **apply_intensity_jitter**: Applies scale and shift to raw channel only
   - **apply_gaussian_blur**: Implements slice-by-slice 2D Gaussian blur
   - **apply_mask_dropout**: Zeros out mask channels randomly
   - **apply_crop_resize**: Implements random crop followed by resize
   - **__call__**: Applies all augmentations in sequence

2. **RandomCubeDataset Class**:
   - **Constructor**: Sets up dataset parameters and loads volumes
   - **_compute_cleft_components**: Pre-computes connected components for mask-aware sampling
   - **_get_random_bbox**: Selects a random bbox volume
   - **_sample_valid_cube**: Extracts a random cube from a volume
   - **_check_component_overlap**: Verifies structure preservation in augmented views
   - **_get_mask_aware_pair**: Implements structure-aware positive pair generation
   - **__len__**: Returns total number of samples (cubes_per_bbox * num_bboxes)
   - **__getitem__**: Returns a pair of augmented views for training

### models/simclr3d.py

This file implements the 3D-adapted SimCLR model:

1. **SwinTransformer3D Class**:
   - Adapts 2D Swin Transformer for 3D data using a slice-by-slice approach
   - Processes each depth slice and aggregates features
   - Adds 3D pooling to combine slice features

2. **ResNet3D Class**:
   - Implements a 3D ResNet model with variable input channels
   - Uses torchvision's video models with adapted first layer
   - Includes fallback to custom implementation if torchvision unavailable
   - Handles creation of residual blocks and layers

3. **ProjectionMLP Class**:
   - Implements projection head following SimCLR paper
   - Two-layer MLP with BatchNorm and ReLU
   - Maps backbone features to lower-dimensional space

4. **SimCLR Class**:
   - Combines backbone and projection head
   - Handles forward pass and feature extraction
   - Supports different backbone architectures

5. **nt_xent_loss Function**:
   - Implements normalized temperature-scaled cross entropy loss
   - Creates positive pairs from differently augmented views
   - Uses cosine similarity with temperature scaling
   - Handles proper masking and loss computation

### train_simclr.py

This file implements the training loop and experiment management:

1. **Optimizer (LARS)**:
   - Layer-wise Adaptive Rate Scaling implementation
   - Adjusts learning rates per layer based on weight norms
   - Special handling for bias and batch norm parameters

2. **Training Utilities**:
   - **setup_distributed**: Configures distributed training
   - **filter_bias_and_bn**: Creates parameter groups for optimizer
   - **get_optimizer_and_scheduler**: Sets up LARS optimizer, cosine scheduler, and gradient scaler
   - **train_epoch**: Implements training loop with mixed precision
   - **save_checkpoint**: Handles model checkpointing

3. **Main Training Loop**:
   - **main_worker**: Core training function handling data, model, and training loop
   - **main**: Entry point parsing arguments and launching training

4. **Command-line Interface**:
   - Comprehensive set of arguments for controlling all aspects of training
   - Data, model, training, distributed, and logging parameters

## Training Process

### Initialization

1. **Data Loading**:
   - Load preprocessed volumes or process from raw data
   - Create dataset with specified parameters
   - Set up data loader with appropriate batch size and workers

2. **Model Setup**:
   - Create SimCLR model with specified backbone
   - Move to appropriate device (CPU/GPU)
   - Convert to DDP model if using distributed training
   - Initialize wandb for experiment tracking

3. **Optimization Setup**:
   - Create LARS optimizer with parameter groups
   - Set up cosine learning rate scheduler
   - Initialize gradient scaler for mixed precision

### Training Loop

1. **Epoch Iteration**:
   - For each epoch, iterate through data loader
   - Optional switch to mask-aware sampling at specified epoch

2. **Forward Pass**:
   - Extract positive pairs of augmented views
   - Forward through model to get projections
   - Compute NT-Xent loss between positive pairs

3. **Backward Pass**:
   - Scale loss for gradient accumulation
   - Backward pass with gradient scaling
   - Gradient clipping to prevent explosion
   - Optimizer step, scaler update, scheduler step

4. **Logging and Checkpointing**:
   - Log metrics to wandb and console
   - Save checkpoints at regular intervals
   - Track best model based on loss

### Finalization

1. **Model Saving**:
   - Save final model and backbone separately
   - Log final artifacts to wandb
   - Close wandb session

## Advanced Features

### Mask-Aware Sampling ("SegCLR Trick")

A key innovation in this implementation is structure-aware sampling:

1. **Component Identification**:
   - Pre-compute connected components in cleft masks
   - Identify significant structural regions

2. **Structure Preservation**:
   - Sample cubes containing significant cleft structures
   - Ensure augmented views preserve these structures
   - Create positive pairs that share biological context

3. **Benefits**:
   - Improves representation quality by focusing on meaningful structures
   - Encourages model to learn structural features rather than noise
   - More efficient use of limited training data

### Mixed Precision Training

Implemented using PyTorch's AMP (Automatic Mixed Precision):

1. **Forward Pass**: Uses float16 precision for speed and memory efficiency
2. **Gradient Scaling**: Prevents underflow in gradients
3. **Optimizer State**: Maintained in float32 for accuracy

### Gradient Accumulation

Enables training with larger effective batch sizes:

1. **Batch Splitting**: Processes smaller batches but accumulates gradients
2. **Delayed Updates**: Updates model only after N mini-batches
3. **Effective Batch Size**: batch_size × gradient_accumulation_steps × num_gpus

### Distributed Training

Supports multi-GPU and multi-node training:

1. **Process Group**: Initializes process group for communication
2. **Data Parallelism**: Each GPU processes different data batches
3. **Gradient Synchronization**: Averages gradients across processes
4. **Batch Normalization**: Uses SyncBatchNorm for consistent statistics

## Logging and Visualization

### Weights & Biases Integration

Comprehensive experiment tracking with wandb:

1. **Metrics**:
   - Training loss at step and epoch level
   - Learning rate tracking
   - Custom metrics for model analysis

2. **Artifacts**:
   - Model checkpoints at regular intervals
   - Best model based on validation metrics
   - Final model and backbone for downstream use

3. **Configuration**:
   - Hyperparameters and training settings
   - Model architecture details
   - Dataset configuration

## Technical Challenges and Solutions

### 3D Data Handling

1. **Memory Constraints**:
   - Challenge: 3D volumes require significant memory
   - Solution: Dynamic sampling of smaller cubes, H5 caching

2. **Efficient Augmentation**:
   - Challenge: 3D augmentations are computationally expensive
   - Solution: Optimized implementations, channel-specific transformations

### Mixed Precision Issues

1. **Numeric Stability**:
   - Challenge: Some operations are sensitive to reduced precision
   - Solution: Explicit float32 conversion for critical operations

2. **PyTorch API Changes**:
   - Challenge: PyTorch's AMP API has evolved
   - Solution: Updated implementation using latest torch.amp module

### Distributed Training

1. **Process Coordination**:
   - Challenge: Ensuring consistent behavior across processes
   - Solution: Proper initialization and synchronization points

2. **Wandb Integration**:
   - Challenge: Avoiding duplicate logging in distributed mode
   - Solution: Only log from rank 0, check for distributed initialization
