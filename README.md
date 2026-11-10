# Skin-Lesion-Segmentation-and-Classification-using-Deep-Learning

# Skin Cancer Detection and Classification Using Deep Learning

A research-oriented deep learning project focused on automated skin-lesion analysis using dermoscopic images. The project explores **lesion segmentation, region-of-interest (ROI) extraction, and binary classification** to investigate how isolating a lesion region can support skin cancer classification.

## Overview

Skin cancer is one of the major concerns in dermatology, and early identification of suspicious skin lesions is important. Dermoscopic images provide detailed visual information about skin lesions, but accurately identifying the lesion region and distinguishing between different diagnostic categories can be challenging.

This project investigates a computer vision pipeline that first identifies the lesion region through image segmentation and then uses the extracted region for classification.

The work is divided into two main stages:

1. **Lesion Segmentation:** Identifying the skin-lesion region and separating it from the surrounding skin.
2. **Lesion Classification:** Analysing the lesion-focused image to classify it into the defined diagnostic categories.

## Project Objectives

- Develop a deep learning-based workflow for skin-lesion analysis.
- Segment lesion regions from dermoscopic images.
- Explore and compare different segmentation architectures and region-localisation techniques.
- Generate lesion-focused images using predicted segmentation masks.
- Train deep learning classifiers for binary classification.
- Evaluate segmentation and classification performance using appropriate metrics.
- Investigate the role of lesion segmentation in improving classification.

## Datasets

This project uses two datasets for different purposes.

### 1. ISIC 2018 – Task 1: Lesion Boundary Segmentation

**Purpose:** Skin-lesion segmentation.

The ISIC 2018 Task 1 dataset provides dermoscopic images along with corresponding ground-truth segmentation masks.

- Input: Dermoscopic skin images.
- Ground truth: Binary lesion masks.
- Task: Pixel-level segmentation.
- Objective: Identify the pixels belonging to the visible lesion region.

The dataset provides image-mask pairs that can be used to train and evaluate segmentation models.

Dataset: [ISIC 2018 Challenge – Task 1](https://challenge.isic-archive.com/landing/2018/45/)

### 2. HAM10000 – Human Against Machine with 10,000 Training Images

**Purpose:** Skin-lesion classification.

HAM10000 contains 10,015 dermoscopic images representing seven diagnostic categories.

| Abbreviation | Diagnostic category |
|---|---|
| nv | Melanocytic nevi |
| mel | Melanoma |
| bkl | Benign keratosis-like lesions |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses and intraepithelial carcinoma |
| vasc | Vascular lesions |
| df | Dermatofibroma |

The dataset includes diagnostic metadata, such as image identifiers, lesion identifiers, diagnosis, age, sex, and anatomical localisation.

For this project, the classification task uses a binary-label setup. The mapping of the original diagnostic categories into the two target classes should be documented according to the project's implemented label mapping.

Dataset: [HAM10000 – Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)

### Why use two different datasets?

The datasets serve complementary purposes:

- ISIC 2018 Task 1 provides ground-truth lesion masks, making it suitable for supervised segmentation.
- HAM10000 provides diagnostic labels, making it suitable for classification.

The segmentation dataset teaches the model **where the lesion is**, while the classification dataset provides information about **the diagnostic category of the lesion**.

## Proposed Methodology

The overall workflow follows a segmentation-to-classification approach.

```text
             Dermoscopic Images
                      |
                      v
              Image Preprocessing
                      |
                      v
             Lesion Segmentation
                      |
          +-----------+-----------+
          |                       |
          v                       v
     U-Net / Other           Superpixel-Based
     Segmentation            Localisation Methods
          |                       |
          +-----------+-----------+
                      |
                      v
              Predicted Lesion Mask
                      |
                      v
             Mask Processing and
              Thresholding
                      |
                      v
             ROI / Lesion-Focused
                   Image
                      |
                      v
              Image Preprocessing
                      |
                      v
             Deep Learning-Based
                Classification
                      |
                      v
               Binary Prediction
```

### Stage 1: Image Preprocessing

The dermoscopic images are prepared before being passed to the segmentation models.

Typical preprocessing operations include:

- Image resizing to a consistent input resolution.
- Pixel-value normalisation.
- Preparation of corresponding ground-truth masks.
- Data augmentation where applicable.

Image and mask transformations must remain spatially aligned.

### Stage 2: Lesion Segmentation

Segmentation models are used to predict the lesion boundary at the pixel level.

The project explores different segmentation approaches, including:

- U-Net
- Attention U-Net
- UNet++
- TransUNet
- Swin-UNet
- nnU-Net

The implemented segmentation experiments also include superpixel-based methods such as:

- Watershed
- SLIC (Simple Linear Iterative Clustering)
- SEEDS (Superpixels Extracted via Energy-Driven Sampling)

These approaches provide different ways of identifying and localising the lesion region.

### Stage 3: Mask Processing and ROI Extraction

The predicted segmentation mask is processed to obtain the lesion region.

The workflow uses mask-based processing and thresholding to distinguish lesion pixels from background pixels.

The processed mask is then used to isolate the corresponding lesion region from the original dermoscopic image.

This generates a **lesion-focused image**, which is subsequently used in the classification stage.

### Stage 4: Binary Classification

The lesion-focused images are passed to deep learning-based classification models.

The classification experiments include:

- ConvNeXt-Tiny
- DenseNet121
- EfficientNetB0

The objective is to classify the input lesion image into the two target categories defined by the project's binary-label mapping.

### Stage 5: Model Evaluation

The segmentation and classification stages are evaluated separately using task-appropriate metrics.

## Evaluation Metrics

### Segmentation Metrics

| Metric | Purpose |
|---|---|
| Dice Coefficient | Measures the overlap between predicted and ground-truth masks. |
| Intersection over Union (IoU) | Measures the intersection relative to the union of the two masks. |

**Dice Coefficient:**

\[
Dice = \frac{2TP}{2TP+FP+FN}
\]

**Intersection over Union:**

\[
IoU = \frac{TP}{TP+FP+FN}
\]

### Classification Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified samples. |
| Precision | Measures how many predicted positive cases are actually positive. |
| Recall / Sensitivity | Measures how many actual positive cases are correctly identified. |
| F1-Score | Harmonic mean of precision and recall. |
| Specificity | Measures how many actual negative cases are correctly identified. |

These metrics provide a broader understanding of model performance, particularly when the dataset contains class imbalance.

## Technologies and Libraries

| Category | Tools / Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV, scikit-image |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Machine Learning | scikit-learn |
| Computer Vision | CNN-based architectures, segmentation models |
| Development Environment | Google Colab / Jupyter Notebook |

## Key Areas of Exploration

- Medical image preprocessing and segmentation.
- CNN-based and Transformer-based segmentation architectures.
- Attention mechanisms and feature extraction.
- Superpixel-based image segmentation.
- Mask thresholding and lesion ROI extraction.
- Deep learning-based binary classification.
- Evaluation of segmentation and classification models.
- Analysis of class imbalance in medical image datasets.

## Repository Structure

The following is a suggested organisation. Update the folder and notebook names to match the files actually uploaded to this repository.

```text
Skin-Cancer-Detection/
│
├── datasets/
│   ├── ISIC2018/
│   └── HAM10000/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── lesion_segmentation.ipynb
│   ├── superpixel_segmentation.ipynb
│   ├── roi_extraction.ipynb
│   └── lesion_classification.ipynb
│
├── models/
│   ├── segmentation/
│   └── classification/
│
├── results/
│   ├── segmentation/
│   └── classification/
│
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If a requirements file is not provided, the principal libraries used in the project include:

```bash
pip install tensorflow opencv-python scikit-image scikit-learn numpy pandas matplotlib
```

Download the ISIC 2018 Task 1 and HAM10000 datasets from their official sources and configure the dataset paths in the notebooks.

## Results

This section should contain the final experimental results after verification.

Recommended result tables:

**Segmentation comparison**

| Model | Dice Coefficient | IoU |
|---|---:|---:|
| Attention U-Net | To be added | To be added |
| nnU-Net | To be added | To be added |
| Other evaluated approaches | To be added | To be added |

**Classification comparison**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | To be added | To be added | To be added | To be added |
| DenseNet121 | To be added | To be added | To be added | To be added |
| EfficientNetB0 | To be added | To be added | To be added | To be added |

Only include models and scores that were actually evaluated. Add a separate comparison of original-image and ROI-based classification only if that experiment was performed.

## Limitations

- Dermoscopic image quality, lighting, and acquisition conditions can affect segmentation performance.
- Skin lesions can have irregular boundaries, making accurate segmentation challenging.
- Class imbalance can influence classification performance.
- Dataset differences may affect model generalisation.
- A segmentation or classification prediction is not equivalent to a clinical diagnosis.

## Future Scope

- Evaluate the pipeline on larger and more diverse dermoscopic datasets.
- Improve generalisation across different image acquisition conditions.
- Explore explainable AI techniques to understand model predictions.
- Investigate class-imbalance handling and class-wise performance.
- Evaluate the effect of lesion ROI extraction on classification performance.
- Explore external validation and clinical evaluation.

## Disclaimer

This project is an academic research and learning exercise in medical image analysis. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical advice or dermatological diagnosis.

## Acknowledgement

This work was carried out as part of a research internship at **NIT Delhi**, focusing on skin-lesion segmentation and classification using deep learning.
