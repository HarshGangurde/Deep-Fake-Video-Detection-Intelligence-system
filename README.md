# 🎭 DeepFake Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![Model](https://img.shields.io/badge/Model-VGG16-red)

An AI-powered **DeepFake Detection Web Application** that analyzes videos using deep learning and computer vision techniques to classify them as **Real or Fake**.

This system leverages **transfer learning (VGG16)**, **face detection**, and **frame-level classification** to detect manipulated media with high accuracy.

---

# 📌 Table of Contents

* Introduction
* Motivation
* Project Objectives
* System Architecture
* Pre-processing Pipeline
* Prediction Pipeline
* Model Architecture
* Hyperparameters
* Model Performance
* Project Structure
* Installation
* Usage
* Technologies Used
* Future Improvements
* License

---

# 📖 Introduction

DeepFake technology uses AI to create synthetic videos that appear highly realistic. These manipulations can be difficult to detect with the human eye.

This project builds a **DeepFake detection pipeline** capable of identifying manipulated videos using deep neural networks and computer vision techniques.

---

# ⚠️ Motivation

DeepFake videos pose serious risks:

* 📰 Spread of misinformation
* 🎭 Identity impersonation
* 🔐 Security vulnerabilities
* 💻 Cyber fraud and scams

This project aims to address these challenges by building a reliable detection system.

---

# 🎯 Project Objectives

* Extract frames from video input
* Detect and crop faces from frames
* Use deep learning for feature extraction
* Classify frames as **Real or Fake**
* Aggregate predictions across frames
* Provide a simple **web interface** for users

---

# 🏗️ System Architecture

```
Video Upload
     │
     ▼
Frame Extraction (OpenCV)
     │
     ▼
Face Detection (MTCNN)
     │
     ▼
Image Preprocessing
     │
     ▼
Feature Extraction (VGG16)
     │
     ▼
CNN Classification
     │
     ▼
Frame Predictions
     │
     ▼
Aggregation
     │
     ▼
Final Output → Real / Fake
```

---

# 🔄 Pre-processing Pipeline

1. Convert video into frames using OpenCV
2. Detect faces using MTCNN
3. Crop facial regions
4. Resize images to **128×128**
5. Normalize pixel values
6. Prepare tensors for model input

---

# 🔍 Prediction Pipeline

1. User uploads a video
2. Frames are extracted
3. Faces are detected and processed
4. Model predicts probability for each frame
5. Predictions are averaged
6. Final result is displayed

**Output Example:**

```
Prediction: Fake
Confidence: 0.94
```

---

# 🧠 Model Architecture

### Feature Extractor

* Pretrained **VGG16** (ImageNet)
* Captures deep visual patterns

### Classification Head

```
VGG16 Base
   │
Global Average Pooling
   │
Dense (512)
   │
Dropout (0.5)
   │
Dense (1)
   │
Sigmoid
```

---

# ⚙️ Hyperparameters

| Parameter     | Value      |
| ------------- | ---------- |
| Optimizer     | Adam       |
| Learning Rate | 0.0001     |
| Loss Function | Focal Loss |
| Batch Size    | 32         |
| Epochs        | 35         |
| Image Size    | 128×128    |


---

# 💻 Installation

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detection-system.git
cd deepfake-detection-system
pip install -r requirements.txt
```

---

# 🚀 Usage

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

Upload a video to detect whether it is **Real or Fake**.

---

# 🛠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* MTCNN
* Flask
* HTML, CSS, JavaScript

---

# 🔮 Future Improvements

* Use **3D CNN / LSTM** for temporal learning
* Improve real-time detection speed
* Increase dataset diversity
* Deploy on cloud platforms
* Add batch video processing


# ⭐ Acknowledgement

This project demonstrates the application of **deep learning in combating AI-generated media manipulation**.

