# Crop Disease Detection

A computer vision system for identifying crop diseases from leaf images using deep learning.

## Overview

This project uses a pretrained ResNet50 model with transfer learning to classify leaf images into **38 crop disease and healthy-leaf classes** across **14 crop types**.

The system includes:

* Image preprocessing and normalization
* Transfer learning with ResNet50
* Multi-class disease classification
* Model training and validation
* Streamlit-based image prediction interface

## Architecture

```text
Leaf Image
    ↓
Resize to 224 × 224
    ↓
Normalize using ImageNet statistics
    ↓
ResNet50 Feature Extractor
    ↓
Fully Connected Classification Layer
    ↓
38 Class Scores
    ↓
Predicted Disease
```

The ResNet50 backbone uses pretrained ImageNet features, while the final classification layer is trained specifically for the plant-disease classes.

## Dataset

This project uses the **New Plant Diseases Dataset (Augmented)**, based on the PlantVillage dataset.

* **~87,000 images**
* **38 classes** - healthy and diseased leaves
* **14 crop types**
* Separate training and validation sets

Class labels are derived automatically from the dataset directory structure using PyTorch's `ImageFolder`.

**Note:** The dataset is not included in this repository due to its large size. It must be downloaded separately.

## Model Training

* **Architecture:** ResNet50
* **Approach:** Transfer learning from ImageNet
* **Backbone:** Frozen pretrained layers
* **Classification head:** Newly trained fully connected layer
* **Loss:** CrossEntropyLoss
* **Optimizer:** Adam
* **Epochs:** 5

Training and validation metrics are tracked throughout the training process.

The original training run achieved **96.3% validation accuracy**.

## Prediction Pipeline

When a user uploads an image through the Streamlit application:

1. The image is loaded and converted to RGB.
2. It is resized to 224×224.
3. ImageNet normalization is applied.
4. The image is passed through the trained ResNet50.
5. The highest-scoring class is selected.
6. The corresponding disease name is displayed.

## Project Structure

```text
crop-disease-detection/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   └── __init__.py
├── test_images/
│   └── sample_leaf.jpg
├── class_names.json
├── app.py
├── test_image.py
├── requirements.txt
├── sample_batch.png
├── training_history.png
└── README.md
```

## Installation & Usage

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

To train the model after downloading and placing the dataset in the expected directory structure:

```bash
python -m src.model
```

**Note:** The trained model weights (`.pth`) are not included in this repository because of their file size.

## Limitations

Performance on real-world field photographs may differ from performance on the curated dataset. Differences in lighting, backgrounds, camera quality, leaf position, and disease appearance can affect predictions.
