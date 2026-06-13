# 🐠 Coral Reef Image Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://coral-reef-classifier.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Overview

**Coral Reef Image Classifier** is a deep learning-based computer vision project developed for the automated classification of coral reef images. The system applies modern convolutional and transformer-based neural network architectures to support coral species identification from underwater imagery.

This project is designed to assist researchers, marine biologists, environmental monitoring teams, and conservation specialists by reducing the manual effort required in reef image analysis and by providing a structured platform for model training, comparison, evaluation, and deployment.

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Develop an automated image classification system for coral reef species.
- Compare multiple state-of-the-art deep learning architectures.
- Build an interactive web application for real-time image inference.
- Evaluate model performance using accuracy, F1-score, and confusion matrices.
- Demonstrate the role of artificial intelligence in marine biodiversity monitoring and coral reef conservation.

---

## 🌟 Key Features

### Multi-Model Deep Learning Framework

The project supports several widely used computer vision architectures, including:

- Vision Transformer (ViT)
- ConvNeXt
- EfficientNet-B0
- ResNet-50
- MobileNet
- DenseNet121

This allows comparative analysis between transformer-based models, modern convolutional neural networks, and lightweight architectures suitable for efficient deployment.

### Interactive Web Application

The project includes a polished **Streamlit** web application that allows users to upload coral reef images and obtain real-time classification results.

The interface supports:

- Image upload
- Model-based prediction
- Prediction confidence display
- Top-K class probability visualization
- User-friendly interaction for research and demonstration purposes

### Model Evaluation and Visualization

The project provides detailed visual evaluation outputs, including:

- Training accuracy curves
- Validation accuracy curves
- Loss curves
- Confusion matrices
- Class-wise performance analysis
- Prediction confidence visualization

These outputs help assess model reliability, generalization ability, and class-level classification performance.

### Modular Codebase

The project is organized in a clear and modular structure, separating:

- Data preparation
- Model training
- Model evaluation
- Saved model weights
- Visualization outputs
- Web application deployment

This structure makes the project easier to maintain, extend, and reproduce.

---

## 🏗️ Project Structure

```text
coral-split/
│
├── app.py                         # Streamlit application for model inference
│
├── notebooks/                     # Model training and experimentation notebooks
│   ├── ConvNeXt.ipynb
│   ├── ViT.ipynb
│   ├── ResNet-50.ipynb
│   ├── MobileNet.ipynb
│   ├── DenseNet121.ipynb
│   └── EfficientNet-B0.ipynb
│
├── Dataset/                       # Dataset split into train, validation, and test sets
│
├── saved_models/                  # Trained model weights
│
├── plots/                         # Training curves and confusion matrices
│
└── README.md                      # Project documentation
```

---

## 🧠 Supported Architectures

| Model | Category | Main Strength | Use Case |
|---|---|---|---|
| **Vision Transformer (ViT)** | Transformer-based model | Captures global image dependencies using attention mechanisms | High-performance image classification |
| **ConvNeXt** | Modern CNN | Strong feature extraction with updated convolutional design | Robust classification tasks |
| **EfficientNet-B0** | Efficient CNN | Balances accuracy and computational efficiency | Efficient model deployment |
| **ResNet-50** | Residual CNN | Stable and reliable baseline architecture | Standard benchmark comparison |
| **MobileNet** | Lightweight CNN | Low computational cost | Mobile and edge deployment |
| **DenseNet121** | Densely connected CNN | Efficient feature reuse and improved gradient flow | Compact and effective classification |

---

## 📊 Dataset and Preprocessing

The models are trained on a curated coral reef image dataset organized into training, validation, and test subsets.

The preprocessing pipeline includes:

- Image resizing to `224 × 224`
- Image normalization using ImageNet statistics
- Data augmentation to improve model generalization
- Batch loading using PyTorch data utilities

Applied augmentation techniques include:

- Random cropping
- Horizontal flipping
- Vertical flipping
- Color jittering
- Perspective transformation

These techniques help improve model robustness against variations commonly found in underwater images, such as lighting changes, water turbidity, image noise, viewpoint differences, and complex reef backgrounds.

---

## ⚙️ Training Configuration

The training workflow follows a standard supervised deep learning image classification pipeline.

| Component | Description |
|---|---|
| Framework | PyTorch |
| Input Image Size | 224 × 224 |
| Loss Function | Cross-Entropy Loss |
| Optimizers | Adam and/or SGD |
| Learning Strategy | Learning rate scheduling |
| Evaluation Metrics | Accuracy, F1-score, and confusion matrix |

Each architecture is trained and evaluated independently to allow fair comparison between models.

---

## 📈 Evaluation

Model performance is evaluated using overall and class-specific metrics.

The evaluation process includes:

- Training accuracy
- Validation accuracy
- Test accuracy
- F1-score
- Confusion matrix analysis
- Prediction confidence visualization
- Top-K class distribution

All evaluation outputs are stored in the `plots/` directory for documentation, analysis, and comparison.

---

## 🚀 Web Application

The project includes a Streamlit application that enables users to interact with the trained models through a graphical interface.

Main application functions include:

- Uploading coral reef images
- Selecting or comparing trained models
- Displaying the predicted coral class
- Showing prediction confidence
- Presenting top-ranked predictions

To run the application locally:

```bash
streamlit run app.py
```

Windows users may also run:

```bash
Run.bat
```

The deployed application is available here:

[Open the Streamlit App](https://coral-reef-classifier.streamlit.app/)

---

## 🛠️ Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/coral-reef-classifier.git
cd coral-reef-classifier
```

### 2. Run the Application

```bash
streamlit run app.py
Or Simpliy Click on Run.bat
```

---

## 🌊 Research and Conservation Relevance

Coral reefs are among the most biodiverse marine ecosystems on Earth. They are highly sensitive to climate change, ocean warming, pollution, bleaching events, and human activity.

Manual analysis of underwater reef imagery is time-consuming and requires expert knowledge. By applying deep learning to coral reef image classification, this project supports:

- Faster reef image analysis
- Improved biodiversity monitoring
- Large-scale ecological survey support
- Early identification of reef health changes
- AI-assisted marine conservation workflows

This project demonstrates how artificial intelligence and computer vision can serve as supportive tools in environmental monitoring and marine science research.

---

## ⚠️ Limitations

Although the system provides a useful automated classification framework, several limitations should be considered:

- Model performance depends on dataset size, diversity, and labeling quality.
- Underwater image quality may affect prediction reliability.
- Similar coral species may be visually difficult to distinguish.
- External validation on independent datasets is recommended before scientific deployment.
- The system should support expert decision-making, not replace domain specialists.

---

## 🔮 Future Work

Potential future improvements include:

- Expanding the dataset with more coral species and reef environments.
- Adding object detection or segmentation for coral localization.
- Improving model explainability using Grad-CAM or attention visualization.
- Deploying lightweight models for mobile or edge devices.
- Integrating geographic and ecological metadata.
- Performing external validation using independent reef monitoring datasets.
- Adding a detailed model comparison report inside the web application.

---

## 🧰 Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL / OpenCV

---

## 📚 Citation

If this project is used for academic, research, or educational purposes, please cite it as:

```text
Coral Reef Image Classifier: A Deep Learning-Based Platform for Automated Coral Reef Image Classification.
Available at: https://github.com/mohamedbadawi81/Coral-Reef-Image-Classifier
```

---

## 📄 Disclaimer

This system is designed for research, educational, and demonstration purposes. For scientific monitoring or conservation decisions, model outputs should be reviewed and validated by qualified marine biology or ecological domain experts.

---

## ❤️ Acknowledgment

This project highlights the potential of artificial intelligence in supporting marine conservation, reef monitoring, and biodiversity research.
