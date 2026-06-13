# 🐠 Coral Reef Image Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://coral-reef-classifier.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

An advanced deep learning platform for identifying coral reef species from underwater imagery. This project leverages state-of-the-art computer vision architectures to assist marine biologists and conservationists in monitoring reef health and biodiversity.

## 🌟 Key Features

- **Multi-Model Architecture**: Support for 7+ modern architectures including **Vision Transformers (ViT)**, **ConvNeXt**, **EfficientNet**, and **ResNet**.
- **Interactive Web Dashboard**: A polished Streamlit interface for real-time inference and model comparison.
- **Explainable AI**: Visualizes prediction confidence and top-K class distributions.
- **Modular Codebase**: Production-ready project structure with separated concerns (data loading, model definitions, training).
- **Comprehensive Evaluation**: Detailed metrics (Accuracy, F1-Score, Confusion Matrices) for every architecture.

## 🏗️ Project Structure

```text
coral-split/
├── app.py                # Streamlit application
├── notebooks/
  ├──ConvNeXt.ipynb
  ├──ViT.ipynb
  ├──ResNet-50.ipynb          
  ├──MobileNet.ipynb
  ├──DenseNet121.ipynb
  ├──EfficientNet-B0.ipynb
├── Dataset/            # Split into train/val/test
├── saved_models/       # Production-ready model weights (.pth)
├── plots/              # Training history & confusion matrices
└── README.md           # Documentation
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/coral-reef-classifier.git
cd coral-reef-classifier
```

### 2. Run the App
```bash
streamlit run app.py
```
Or simply use the provided `Run.bat` on Windows.

## 🧠 Supported Architectures

| Model | Description | Use Case |
| :--- | :--- | :--- |
| **ViT (Vision Transformer)** | Attention-based architecture | High precision on large datasets |
| **ConvNeXt** | Modernized CNN | State-of-the-art accuracy |
| **EfficientNet-B0** | Compound-scaled CNN | Optimal performance/efficiency balance |
| **ResNet-50** | Residual Learning | Industry standard baseline |
| **MobileNetV2/V3** | Lightweight CNNs | Mobile & Edge deployment |
| **DenseNet121** | Densely connected layers | Feature reuse & efficiency |

## 📊 Dataset & Training

The models are trained on a curated dataset of coral reef images, categorized into various species. 
- **Preprocessing**: 224x224 resizing, ImageNet normalization.
- **Augmentations**: Random cropping, horizontal/vertical flips, color jitter, and perspective transforms.
- **Optimizer**: Adam/SGD with Learning Rate Scheduling.
- **Loss Function**: Cross-Entropy Loss.

## 📈 Results

Each model's performance is documented in the `plots/` directory. 
- **Accuracy Curves**: Visualize convergence and stability.
- **Confusion Matrices**: Detailed breakdown of class-wise performance.

Example results:
- **ViT** achieves high accuracy by capturing global dependencies.
- **ConvNeXt** provides robust feature extraction with a convolutional inductive bias.

## 🌊 Conservation Impact

Coral reefs are the "rainforests of the sea." By automating the identification process, this tool reduces the manual labor required for reef surveys, enabling faster response times to bleaching events and biodiversity shifts.

---
*Made with ❤️*
"# Coral-Reef-Image-Classifier" 
"# Coral-Reef-Image-Classifier" 
