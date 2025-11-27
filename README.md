# GastroVision: AI-Powered Gastrointestinal Disease Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-96.11%25-success)

Deep learning system for automated classification of gastrointestinal conditions from endoscopy images.

## 🏆 Performance

- **Best Model:** Ensemble (EfficientNet-B3 + Vision Transformer)
- **Validation Accuracy:** 96.11%
- **Classes:** 4 (Colon polyps, Erythema, Normal esophagus, Normal mucosa)

## 🚀 Quick Start

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))

### Installation

1. Clone repository:
```bash
git clone <your-repo-url>
cd GastroVision_Docker
```

2. Build Docker image:
```bash
docker build -t gastrovision:latest .
```

### Usage

**Run prediction on single image:**

Mac/Linux:
```bash
./run_prediction.sh test_images/Colon_polyps/image.jpg
```

Windows:
```batch
run_prediction.bat test_images\Colon_polyps\image.jpg
```

**Manual Docker command:**
```bash
docker run --rm \
    -v "$(pwd)/test_images:/app/test_images" \
    gastrovision:latest \
    python src/predict.py --image test_images/YOUR_IMAGE.jpg --models_dir models/
```

## 📊 Model Architecture

### Ensemble Approach
- **Model 1:** EfficientNet-B3 (94.27% accuracy)
  - Weight: 40%
  - Strength: Efficient CNN architecture
  
- **Model 2:** Vision Transformer (95.50% accuracy)
  - Weight: 60%
  - Strength: Global attention mechanism

### Training Details
- **Dataset:** GastroVision (2,442 images)
- **Split:** 80% train, 20% validation
- **Augmentation:** Rotation, flipping, color jitter
- **Loss:** Weighted Cross-Entropy (handles class imbalance)
- **Optimizer:** Adam with learning rate scheduling

## 📁 Project Structure
```
GastroVision_Docker/
├── models/                          # Trained model weights
│   ├── best_efficientnet_focal.pth
│   ├── best_vit_model.pth
│   └── class_mapping.json
├── src/
│   └── predict.py                   # Inference script
├── test_images/                     # Sample test images
├── Dockerfile                       # Docker configuration
├── requirements.txt                 # Python dependencies
├── run_prediction.sh               # Easy prediction script (Unix)
├── run_prediction.bat              # Easy prediction script (Windows)
└── README.md                       # This file
```

## 🔬 Results

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Colon polyps | 96.1% | 90.2% | 93.8% | 164 |
| Erythema | 100.0% | 33.3% | 50.0% | 3 |
| Normal esophagus | 100.0% | 100.0% | 100.0% | 28 |
| Normal mucosa | 94.7% | 98.0% | 96.3% | 294 |

**Overall Accuracy:** 96.11%

### Model Comparison

| Model | Accuracy |
|-------|----------|
| ResNet-50 (Baseline) | 68.92% |
| EfficientNet-B3 | 94.27% |
| EfficientNet-B4 | 93.66% |
| Vision Transformer | 95.50% |
| **Ensemble (Final)** | **96.11%** |

## 🎓 Technical Details

### Interpretability
- Grad-CAM visualizations show model focuses on clinically relevant regions
- Polyps: Localized attention on lesion areas
- Erythema: Diffuse attention on inflamed tissue

### Validation
- Stability: 0% variance across runs
- Calibration: 2.7% error (excellent)
- Model is well-calibrated and reliable

## 👨‍💻 Author

**Nishant Joshi**
- University of South Dakota
- Machine Learning Fundamentals - Final Project
- Date: December 2024

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset: GastroVision Challenge
- Frameworks: PyTorch, timm
- Inspiration: Clinical AI applications in gastroenterology
