# TrackNet Enhanced PyTorch

A PyTorch implementation of TrackNet with architectural enhancements for sports object tracking, specifically optimized for badminton shuttlecock detection and tracking.

## Overview

TrackNet Enhanced is a deep learning framework designed for precise object tracking in sports videos. This implementation extends the original TrackNet architecture with temporal modeling capabilities using GRU layers and enhanced feature extraction for improved tracking accuracy.

### Key Features

- **Enhanced Architecture**: U-Net style encoder-decoder with GRU temporal modeling
- **Multi-frame Input**: Processes 5 consecutive frames (15 channels) for temporal context
- **Multi-target Output**: Generates 3-channel heatmaps for robust detection
- **Flexible Training**: Support for multiple optimizers and learning rate scheduling
- **Efficient Data Pipeline**: HDF5-based dataset format for fast loading
- **Comprehensive Evaluation**: Detailed accuracy metrics and performance analysis

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+ 
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/AnInsomniacy/tracknet-enhanced-pytorch.git
cd tracknet-enhanced-pytorch

# Install required packages
pip install torch torchvision opencv-python h5py tqdm matplotlib seaborn pandas numpy scipy
```

## Usage

### Data Preprocessing

Convert video datasets to HDF5 format for training:

```bash
python preprocess/preprocess_to_hdf5.py --source dataset_path --output dataset.h5
```

### Training

Train the enhanced TrackNet model:

```bash
# Basic training
python train.py --train train_data.h5 --val val_data.h5

# Advanced training with custom parameters
python train.py --train train_data.h5 --val val_data.h5 \
  --batch 8 --epochs 100 --optimizer Adam --lr 0.001 \
  --scheduler ReduceLROnPlateau --out outputs --name experiment
```

### Evaluation

Test model performance on validation data:

```bash
# Basic evaluation
python test.py --model best_model.pth --data test_data.h5

# Detailed evaluation with custom threshold
python test.py --model best_model.pth --data test_data.h5 \
  --threshold 0.3 --tolerance 4 --report detailed
```

### Video Prediction

Apply trained model to video files:

```bash
python predict/video_predict.py --model best_model.pth --input video.mp4 --output result.mp4
```

## Architecture

The enhanced TrackNet architecture features:

- **Input**: 15-channel tensor (5 frames × 3 RGB channels) at 512×288 resolution
- **Encoder**: VGG-style feature extraction with batch normalization
- **Temporal Modeling**: GRU layer for sequential frame relationships  
- **Decoder**: Skip-connected upsampling with feature fusion
- **Output**: 3-channel heatmap for multi-target detection

## Dataset Format

The model expects HDF5 datasets with the following structure:

```
dataset.h5
├── images: [N, 15, 288, 512] - Input frame sequences
└── labels: [N, 3, 288, 512] - Target heatmaps
```

- **N**: Number of samples
- **15 channels**: 5 consecutive frames (t-2, t-1, t, t+1, t+2)
- **3 output channels**: Multiple heatmap targets for robust detection

## Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--batch` | Batch size | 3 |
| `--epochs` | Training epochs | 30 |
| `--optimizer` | Optimizer (Adam/AdamW/SGD/Adadelta) | Adadelta |
| `--lr` | Learning rate | Auto |
| `--scheduler` | LR scheduler | ReduceLROnPlateau |
| `--threshold` | Detection threshold | 0.5 |

## Performance

The enhanced architecture provides improved tracking accuracy through:

- Temporal context modeling with GRU layers
- Multi-scale feature extraction
- Skip connections for detail preservation
- Robust multi-target heatmap generation

## License

MIT License - see LICENSE file for details.
