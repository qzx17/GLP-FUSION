# GLP-Fusion: Multi-modal Student Engagement Recognition System

A deep learning-based multi-modal fusion framework for automatic student engagement recognition. This project combines visual features (TemporalViT), behavioral features (temporal-spatial stream), and physiological signals (SL features) through a pyramid fusion module to achieve high-precision engagement classification.

## Table of Contents
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Parameter Description](#parameter-description)

## Introduction

This project implements a multi-modal feature fusion-based student engagement recognition system, supporting the following datasets:
- **DAiSEE**: Dataset for Affective States in E-learning Environments
- **EngageNet**: Emotion Recognition in the Wild

### Key Features
- 🎯 Multi-modal Fusion: Visual + Behavioral + Physiological Signals
- 🧠 Transformer-based Temporal-Spatial Feature Extraction
- 🔥 Pyramid Feature Fusion Architecture
- 📊 Complete Training and Evaluation Pipeline
- 🚀 GPU-accelerated Training Support

### Classification Categories
The system classifies student engagement into 4 levels:
1. Not-Engaged
2. Barely-Engaged
3. Engaged
4. Highly-Engaged

## System Architecture

### Model Components

```
Input Data
├── Video Frame Sequence (16 frames × 3 channels × 224×224)
├── Behavioral Features (pose, gaze)
└── Physiological Signals (heart rate, peak intervals)
        ↓
Feature Extraction
├── TemporalViT (Visual Feature Extraction)
│   ├── CNN Preprocessing Layer
│   ├── Residual Block
│   ├── Patch Embedding
│   └── Temporal Transformer
├── TS_Stream (Behavioral Feature Extraction)
│   ├── Temporal-Spatial Convolution Module
│   └── Transformer Encoder
└── SLEngagementNet (Physiological Signal Processing)
    └── Dense Connection Network
        ↓
Feature Fusion
└── PyramidFusionModule (Pyramid Fusion)
    ├── Multi-scale Transformer (Low/Mid/High Resolution)
    ├── Cross-scale Feature Fusion
    └── Temporal Position Encoding
        ↓
Classification Output
└── EnhancedFusion
    ├── Attention Weighting
    └── 4-class Engagement Output
```

### Core Technologies
1. **TemporalViT**: Vision Transformer-based temporal visual feature extraction
2. **TS_Stream**: Temporal-spatial stream for behavioral features (pose, gaze)
3. **PyramidFusion**: Pyramid-style multi-scale feature fusion
4. **SLEngagementNet**: Physiological signal feature encoding

## Environment Setup

### System Requirements
- Python 3.8
- CUDA 11.8 (GPU recommended)
- 16GB+ GPU Memory (recommended)

### Installation Steps

1. **Clone the Repository**
```bash
git clone <your-repo-url>
cd GLP-fusion
```

2. **Create Virtual Environment**
```bash
# Using conda
conda create -n glp-fusion python=3.8
conda activate glp-fusion

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install PyTorch (based on your CUDA version)**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU version
pip install torch torchvision torchaudio
```

## Data Preparation

### Dataset Structure

You need to email abhinav@iitrpr.ac.in or  abhinav@iitrpr.ac.in to apply for and obtain the EngageNet dataset.
https://github.com/engagenet/engagenet_baselines
You need to download DAiSEE dataset from this web
https://people.iith.ac.in/vineethnb/resources/daisee/index.html

We need to filter out video data with a duration of fewer than 280 frames, videos without students, and extracted black images (as noisy samples) !

```
GLP-fusion/
├── data/
│   ├── DAiSEE/
│   │   ├── frames/          # Video frames
│   │   │   ├── video_001/
│   │   │   │   ├── 001.jpg
│   │   │   │   ├── 002.jpg
│   │   │   │   └── ...
│   │   ├── features/        # Behavioral feature CSV files
│   │   │   ├── video_001.csv
│   │   │   └── ...
│   │   ├── DAiSEE_Train_set.txt
│   │   ├── DAiSEE_Validation_set.txt
│   │   ├── DAiSEE_Test_set.txt
│   │   └── SL_DAiSEE.csv
│   └── EngageNet/
│       └── (similar structure)
```

### Annotation File Format

**Train/Validation/Test Set (.txt)**
```
<video_frame_path> <num_frames> <label> <feature_csv_path>
/path/to/frames/video_001 256 2 /path/to/features/video_001.csv
/path/to/frames/video_002 256 1 /path/to/features/video_002.csv
```

**Behavioral Features CSV**
- Required columns: `pose_Rx`, `pose_Ry`, `gaze_0_x`, `gaze_0_y`, `gaze_1_x`, `gaze_1_y`

**Physiological Signals CSV (SL file)**
- Required columns: `all_ids`, `heart_rates`, `p2p_intervals`, `sys_peaks`, `dys_peaks`

## Model Training

### Basic Training Commands

**Train on DAiSEE Dataset**
```bash
python main_features.py \
  --dataset DAiSEE \
  --workers 8 \
  --epochs 50 \
  --batch-size 6 \
  --lr 0.005 \
  --weight-decay 0.0001 \
  --momentum 0.9 \
  --print-freq 10 \
  --milestones 40 \
  --exper-name "DAiSEE_Experiment" \
  --seed 42 \
  --temporal-layers 1
```

**Train on EngageNet Dataset**
```bash
python main_features.py \
  --dataset EngageNet \
  --workers 8 \
  --epochs 50 \
  --batch-size 6 \
  --lr 0.01 \
  --weight-decay 0.0001 \
  --momentum 0.9 \
  --print-freq 10 \
  --milestones 40 \
  --exper-name "EngageNet_Experiment" \
  --seed 42 \
  --temporal-layers 1
```

### Training Output

The training process will automatically generate the following files:
```
/log/
├── <dataset>-<timestamp>-set1-log.txt      # Training log
├── <dataset>-<timestamp>-set1-log.png      # Training curve
└── <dataset>-<timestamp>-set1-cn.png       # Confusion matrix

/pth/<dataset>/
├── <dataset>-<timestamp>-set1-model.pth       # Latest model
└── <dataset>-<timestamp>-set1-model_best.pth  # Best model
```

## Model Inference

### Inference Script

Edit the `inference.py` file to set test data paths and model path:

```python
# Set test data
test_annotation_file_path = "/EngageNet_Test_set.txt"
sl_file_path = "SL_EngageNet.csv"

# Load model
model = load_model('/pth/EngageNet/EngageNet-xxx-set1-model_best.pth')
```


Run inference:
```bash
python inference.py
```

### Inference Output

```
/result/
├── confusion_matrix.png           # Confusion matrix visualization
└── evaluation_results.txt         # Detailed evaluation metrics
```

**Evaluation metrics include:**
- Weighted Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Correlation Coefficient
- Mean Absolute Error
- Per-class metrics

### Evaluation Metrics

- **UAR** (Unweighted Average Recall): Unweighted average recall
- **WAR** (Weighted Average Recall): Weighted average recall
- **Accuracy**: Accuracy
- **Precision**: Precision
- **Recall**: Recall
- **F1-Score**: F1 score
- **Confusion Matrix**: Confusion matrix

## Parameter Description

### Data Augmentation Parameters

- `GroupRandomSizedCrop`: Random crop to 0.08-1.0 of original size
- `GroupRandomHorizontalFlip`: 50% probability horizontal flip
- `GroupNormalize`: Dataset-specific normalization

### Model Hyperparameters

**TemporalViT:**
- `image_size`: (224, 224)
- `patch_size`: 16
- `dim`: 768
- `depth`: 6 (Number of Transformer layers)
- `heads`: 12 (Number of attention heads)
- `dropout`: 0.3

**PyramidFusion:**
- `depths`: [2, 2, 2] (Transformer depth at each scale)
- `heads`: [2, 2, 2] (Attention heads at each scale)
- `dims`: [64, 32, 16] (Feature dimensions at each scale)

## Important Notes

1. **Data Paths**: Modify data paths in `main_features.py` and `inference.py` according to your setup
2. **GPU Memory**: If GPU memory is insufficient, reduce `batch-size`
3. **Model Dependencies**: Ensure `models/Text.py` file exists (contains class name definitions)
4. **Random Seed**: Set fixed random seed for reproducible results

## FAQ

**Q: How to modify the number of classes?**

Modify the parameter in `Decision_Fusion(n_classes=4)`

**Q: How to use a custom dataset?**

1. Prepare annotation files in the required format
2. Add dataset configuration in `main_features.py`
3. Update normalization parameters in the data loader

**Q: Slow inference speed?**
 
1. Use GPU for inference
2. Reduce num_segments (number of video frames)
3. Use model quantization or pruning

## Citation

If this project helps your research, please cite:
```bibtex
@article{glp-fusion,
  title={GLP-Fusion: Multi-modal Student Engagement Recognition},
  author={Zixuan Qin, Shuanghong Shen, Dengdi Sun, Junyu Lu, Keyu Zhu, Zhenya Huang, Qi Liu},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or suggestions, please contact:
- Email: wa24201053.ahu@vip.163.com

---

**Happy Coding!** 🎓📚








