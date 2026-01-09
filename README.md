# 🎯 Face Detection System - Classical Computer Vision

A robust face detection system using **HOG (Histogram of Oriented Gradients)** and **LBP (Local Binary Patterns)** for feature extraction, combined with **SVM (Support Vector Machine)** classification. This project demonstrates the effectiveness of classical computer vision techniques for face detection tasks.

## 📊 Performance Metrics

- **Precision**: 69.94%
- **Recall**: 58.46%
- **F1-Score**: 63.69%
- **Dataset**: Human Faces Object Detection (Kaggle)
- **Test Set**: 100 images, 195 ground truth faces

## 🌟 Key Features

- ✅ **Classical Computer Vision**: HOG + LBP feature extraction (no deep learning)
- ✅ **Multi-Scale Detection**: Pyramid search for scale-invariant face detection
- ✅ **Hard Negative Mining**: Iterative training to reduce false positives
- ✅ **Non-Maximum Suppression**: Eliminates duplicate detections
- ✅ **Comprehensive Evaluation**: Precision, recall, F1-score with color-coded visualizations
- ✅ **User-Friendly GUI**: Intuitive interface for training, detection, and evaluation
- ✅ **Batch Processing**: Detect faces in multiple images simultaneously
- ✅ **Detailed Logging**: Real-time progress tracking and metrics

## 📁 Project Structure

```
face-detection-classical/
├── config.py                 # Configuration and hyperparameters
├── csv_parser.py            # Dataset annotation parser
├── feature_extraction.py    # HOG and LBP implementation
├── training.py              # SVM training with hard negative mining
├── detector.py              # Multi-scale face detection
├── evaluator.py             # Evaluation framework with metrics
├── main.py                  # GUI application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/
│   └── faces_dataset/      # Dataset (not included)
│       ├── train/
│       │   ├── images/
│       │   └── annotations.csv
│       └── test/
│           ├── images/
│           └── annotations.csv
├── models/
│   └── face_detector.pkl   # Trained model (generated)
└── outputs/
    ├── detections/         # Detection results
    └── evaluation_results/ # Evaluation visualizations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/face-detection-classical.git
cd face-detection-classical
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**

Download the [Human Faces Object Detection dataset](https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection) from Kaggle and organize as:

```
data/faces_dataset/
├── train/
│   ├── images/        # Training images
│   └── annotations.csv
└── test/
    ├── images/        # Test images
    └── annotations.csv
```

### Usage

#### Option 1: GUI Application (Recommended)

```bash
python main.py
```

The GUI provides:
- **View Config**: Display hyperparameters
- **Dataset Info**: Show dataset statistics
- **Train Model**: Interactive training with progress tracking
- **Detect Single Image**: Detect faces in a single image
- **Batch Detection**: Process multiple images
- **Evaluate Model**: Run comprehensive evaluation
- **Open Output Folder**: Quick access to results

#### Option 2: Command Line

**Train the model:**
```bash
python training.py
```

**Detect faces in test images:**
```bash
python detector.py
```

**Evaluate on test set:**
```bash
python evaluator.py
```

## ⚙️ Configuration

Key parameters in `config.py`:

```python
# Feature Extraction
WINDOW_SIZE = (64, 64)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_POINTS = 24
LBP_RADIUS = 3

# Classification
SVM_KERNEL = 'rbf'
SVM_C = 5.0

# Detection
DETECTION_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.3
SCALE_FACTOR = 1.2
SLIDE_STEP = 12

# Training
MAX_TRAIN_IMAGES = 400
HARD_NEGATIVE_MINING = True
HARD_NEGATIVE_ROUNDS = 2
```

## 🧠 How It Works

### 1. Feature Extraction

**HOG (Histogram of Oriented Gradients):**
- Captures shape and edge information
- Computes gradient magnitude and orientation
- Creates cell-based histograms with bilinear interpolation
- Normalizes over 2×2 cell blocks using L2-Hys
- Produces 1764-dimensional feature vector

**LBP (Local Binary Patterns):**
- Encodes local texture patterns
- Compares each pixel with circular neighbors
- Uses uniform patterns for dimensionality reduction
- Produces 555-dimensional feature vector

**Combined Feature Vector:** 2319 dimensions (1764 HOG + 555 LBP)

### 2. Training Pipeline

1. **Data Collection**: Extract positive samples (faces) and negative samples (non-faces)
2. **Data Augmentation**: Horizontal flipping of face samples
3. **Initial Training**: Train RBF-SVM classifier
4. **Hard Negative Mining** (2 rounds):
   - Run detector on training images
   - Collect false positives as hard negatives
   - Retrain classifier with augmented dataset
5. **Model Persistence**: Save trained SVM to pickle file

### 3. Multi-Scale Detection

1. **Image Preprocessing**: Resize large images for efficiency
2. **Pyramid Construction**: Scale image from 1.0× to 0.5× with factor 1.2
3. **Sliding Window**: 64×64 window with 12-pixel stride
4. **Classification**: Extract features and compute SVM decision function
5. **Post-Processing**: Non-Maximum Suppression (NMS) to eliminate duplicates

### 4. Evaluation

- **IoU-based Matching**: Match detections to ground truth (threshold: 0.35)
- **Metrics Computation**: Precision, recall, F1-score
- **Visualization**: Color-coded results (TP=green, FP=orange, FN=red, GT=blue)

## 📊 Detection Results

Evaluated on 100 test images with 195 ground truth faces:

| Metric | Value |
|--------|-------|
| Detected Faces | 163 |
| True Positives | 114 |
| False Positives | 49 |
| False Negatives | 81 |
| **Precision** | **69.94%** |
| **Recall** | **58.46%** |
| **F1-Score** | **63.69%** |

### Strengths
- ✅ Robust frontal face detection
- ✅ Handles illumination variations
- ✅ Multi-face detection in group photos
- ✅ Scale-invariant through pyramid search
- ✅ Effective background clutter rejection

### Limitations
- ❌ Profile faces (side views)
- ❌ Severe occlusions
- ❌ Very small faces (<30×30 pixels)
- ❌ Extreme lighting conditions
- ❌ Motion blur

## 👤 Author

**Violeta Maria Hoza**
- University: Technical University of Cluj-Napoca
- Course: Pattern Recognition Systems
- Year: 2025-2026

## 🙏 Acknowledgments

- Dataset: [Human Faces Object Detection (Kaggle)](https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection)


**⭐ Star this repository if you found it helpful!**
