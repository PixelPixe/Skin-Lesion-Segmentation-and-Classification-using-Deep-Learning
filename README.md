# Skin Lesion Segmentation and Classification Using Deep Learning

A two-stage deep learning framework for skin-lesion analysis using dermoscopic images, combining **lesion segmentation and transfer-learning-based classification**.

**Research Internship:** National Institute of Technology Delhi (NIT Delhi)  
**Duration:** 1 June – 26 June 2026  
**Institution:** Manipal University Jaipur  
**Supervisor:** Dr. Gautam Kumar, Assistant Professor, Department of CSE, NIT Delhi

---

## 1. Project Overview

Skin cancer detection is an important research area because early identification of suspicious lesions can support timely medical evaluation.

This project explores a deep learning-based approach to skin-lesion analysis. It investigates whether focusing on the lesion region can provide useful information for classifying skin lesions.

The project follows two main stages:

1. **Segmentation:** Identify the lesion region from dermoscopic images.
2. **Classification:** Use lesion-focused images to classify lesions into Nevus and Melanoma categories.

The main research question was:

> Can lesion-focused image classification provide useful results compared with ordinary image classification?

The project investigates this question through segmentation and classification experiments. However, an unsegmented-image baseline was not conducted, so the improvement attributable to segmentation has not been established.

## 2. Project Objectives

- Segment skin lesions from dermoscopic images.
- Compare six segmentation architectures.
- Generate lesion-focused images using predicted segmentation masks.
- Classify lesions into Nevus and Melanoma categories.
- Evaluate segmentation and classification performance.
- Investigate the use of segmentation masks in the classification workflow.

## 3. Datasets

Two datasets were used for different stages of the project.

### A. ISIC 2018 – Task 1

**Purpose:** Lesion boundary segmentation.

ISIC 2018 Task 1 provides dermoscopic RGB images and corresponding binary ground-truth lesion masks.

| Dataset split | Number of images |
|---|---:|
| Training | 2,594 |
| Validation | 100 |
| Test | 1,000 |

**Key characteristics:**
- Dermoscopic RGB images.
- Binary ground-truth segmentation masks.
- Variation in lesion sizes, illumination and skin tones.
- Used to train and evaluate lesion segmentation architectures.

Dataset: [ISIC 2018 Challenge – Task 1](https://challenge.isic-archive.com/landing/2018/45/)

### B. HAM10000

**Purpose:** Binary lesion classification.

HAM10000 originally contains 10,015 dermoscopic images across seven diagnostic categories. This project uses a selected binary subset containing Nevus and Melanoma images.

| Class | Label | Number of images |
|---|---:|---:|
| Nevus | 0 | 6,705 |
| Melanoma | 1 | 1,113 |
| **Total** | | **7,818** |

The selected subset focuses on distinguishing melanocytic nevi from melanoma.

**Dataset characteristics:**
- Dermoscopic skin-lesion images.
- Binary classification labels.
- Imbalanced class distribution.
- Used for training and evaluating transfer-learning-based classifiers.

Dataset: [HAM10000 – Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)

### Why were two datasets used?

The datasets serve different purposes:

- ISIC 2018 Task 1 provides image-mask pairs for supervised lesion segmentation.
- HAM10000 provides diagnostic labels for lesion classification.

In simple terms, ISIC 2018 is used to identify **where the lesion is**, while HAM10000 is used to investigate **which diagnostic category the lesion belongs to**.

## 4. Proposed Methodology

The project follows two interconnected experimental workflows.

### Stage 1: Skin Lesion Segmentation

The first stage focuses on identifying lesion boundaries in dermoscopic images using ISIC 2018 Task 1.

**Workflow:**

```text
Dermoscopic Image
       |
       v
Image Preprocessing
(Resizing and Normalisation)
       |
       v
Train Six Segmentation Models
       |
       v
Predicted Binary Mask
       |
       v
Evaluation Against Ground Truth
       |
       v
Dice Coefficient and IoU
```

#### Segmentation Architectures

Six segmentation architectures were compared:

| Architecture | Main concept |
|---|---|
| U-Net | Encoder-decoder architecture with skip connections. |
| U-Net++ | Nested dense skip connections to improve feature fusion. |
| Attention U-Net | Attention gates in skip connections to emphasise relevant regions. |
| ResUNet | Residual blocks for improved gradient flow and feature reuse. |
| TransAttUNet | Transformer-style attention for local and global context. |
| nnU-Net | Self-configuring framework that adapts preprocessing, architecture and training. |

### Stage 2: Lesion-Focused Image Classification

The second stage uses segmentation masks to isolate lesion regions before classification.

Two classification workflows were explored:

**A. HAM10000 classification using U-Net masks**

```text
HAM10000 Dermoscopic Image
          |
          v
Resize to 256 × 256
          |
          v
U-Net Predicted Mask
          |
          v
Mask Thresholding
          |
          v
Mask × Original Image
          |
          v
Lesion-Focused Image
          |
          v
CNN-Based Classification
          |
          v
Nevus (0) / Melanoma (1)
```

**B. ISIC 2018 classification using TransAttUNet masks**

The predicted masks from TransAttUNet were used to generate lesion-focused images, which were then passed to the classification models.

#### Understanding Mask Processing

The predicted segmentation mask is thresholded to obtain a binary representation of the lesion region.

The binary mask is multiplied element-wise with the original image:

\[
I_{\text{ROI}} = I_{\text{original}} \times M_{\text{binary}}
\]

Where:

- \(I_{\text{ROI}}\) = lesion-focused image.
- \(I_{\text{original}}\) = original dermoscopic image.
- \(M_{\text{binary}}\) = predicted binary lesion mask.

This operation retains the pixels inside the predicted lesion region and suppresses pixels outside it.

## 5. Classification Architectures

Transfer learning was used to train CNN-based classifiers with pretrained backbones.

The following architectures were evaluated:

| Model | Main characteristic |
|---|---|
| DenseNet121 | Dense connectivity and feature reuse. |
| MobileNetV2 | Lightweight architecture using depthwise separable convolutions. |
| ResNet50 | Residual connections for effective feature learning. |
| Inception | Parallel convolution operations for multi-scale feature extraction. |
| ConvNeXt-Tiny | Modernised CNN architecture with larger convolution kernels. |

ConvNeXt-Tiny was evaluated on the ISIC 2018 classification workflow, while the other four classifiers were evaluated on both datasets.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Input image size | 224 × 224 |
| Optimizer | Adam |
| Learning rate | 0.00001 (1e-5) |
| Loss function | Binary Cross-Entropy |
| Augmentation | Rotation, horizontal flip, vertical flip and zoom |
| Evaluation metrics | Accuracy, AUC, Precision, Recall and F1-Score |

## 6. Evaluation Metrics

### Segmentation Metrics

**Dice Coefficient**

Measures the overlap between the predicted mask and the ground-truth mask.

\[
Dice = \frac{2|A\cap B|}{|A|+|B|}
\]

**Intersection over Union (IoU)**

Measures the intersection of the predicted and ground-truth masks relative to their union.

\[
IoU = \frac{|A\cap B|}{|A\cup B|}
\]

Both metrics range from 0 to 1, where 1 represents perfect overlap.

### Classification Metrics

- **Accuracy:** Proportion of correctly classified images.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **AUC:** Measures the ability of the classifier to distinguish between the two classes across classification thresholds.

## 7. Experimental Results

### A. Segmentation Results – ISIC 2018

The following values are reported in the presentation's segmentation comparison table.

| Model | Validation Accuracy | Dice | IoU |
|---|---:|---:|---:|
| U-Net | 94.50% | 0.8492 | 0.7674 |
| U-Net++ | 94.95% | 0.8815 | 0.7881 |
| nnU-Net | 94.59% | 0.8720 | 0.7731 |
| Attention U-Net | 94.36% | 0.8800 | 0.7660 |
| TransAttUNet | 94.91% | 0.8810 | 0.7874 |
| ResUNet | 95.36% | 0.8911 | 0.8036 |

**Observation:** ResUNet achieved the highest Dice and IoU values in the reported comparison table.

*Note: The presentation also reports different ResUNet values in Figure 5.12 (approximately 0.9149 Dice and 0.8431 IoU). This discrepancy needs verification against the original experiment logs before treating either result as final.*

### B. Classification Results – Segmented HAM10000

The following results are reported for the HAM10000 classification workflow using U-Net masks.

| Model | Accuracy | Melanoma Recall |
|---|---:|---:|
| DenseNet121 | 0.85 | 0.90 |
| MobileNetV2 | 0.83 | 0.86 |
| ResNet50 | 0.76 | 0.76 |
| Inception | 0.75 | 0.77 |

**Observation:** DenseNet121 achieved the highest reported accuracy (0.85) and melanoma recall (0.90).

*Note: The Inception result was evaluated using a different, imbalanced test set and is not directly comparable to the other models.*

### C. Classification Results – Segmented ISIC 2018

The following results are reported using TransAttUNet masks.

| Model | Accuracy | Melanoma Recall | Melanoma F1 |
|---|---:|---:|---:|
| DenseNet121 | 0.84 | 0.59 | 0.58 |
| MobileNetV2 | 0.84 | 0.74 | 0.64 |
| ResNet50 | 0.80 | 0.66 | 0.56 |
| Inception | 0.78 | 0.77 | 0.57 |
| ConvNeXt-Tiny | 0.81 | 0.02 | 0.03 |

**Observation:**
- DenseNet121 and MobileNetV2 achieved the highest reported accuracy of 0.84.
- MobileNetV2 achieved a higher melanoma recall and melanoma F1-score than DenseNet121.
- ConvNeXt-Tiny showed very poor melanoma detection despite its reported accuracy.

## 8. Key Findings

- Six segmentation architectures were compared on ISIC 2018 Task 1.
- ResUNet achieved the highest reported Dice and IoU in the main comparison table.
- DenseNet121 achieved 0.85 accuracy on the segmented HAM10000 dataset.
- On segmented ISIC 2018, DenseNet121 and MobileNetV2 achieved 0.84 accuracy.
- Classification performance varied across architectures, particularly for melanoma recall.
- The effect of segmentation on classification accuracy remains unverified because an unsegmented-image baseline was not conducted.

## 9. Limitations

- No unsegmented-image baseline was conducted to measure the direct effect of segmentation on classification.
- The datasets have class imbalance, which can affect classification performance.
- Some classifiers showed weak melanoma detection.
- The Inception HAM10000 result used a different test set, limiting direct comparison.
- The project does not include a deployed graphical user interface or computer-aided diagnostic application.
- The models have not undergone clinical validation.

## 10. Future Scope

- Explore advanced attention-based and Transformer-based segmentation architectures.
- Evaluate models on larger and more diverse datasets covering different skin tones and image acquisition conditions.
- Develop an integrated web or desktop application.
- Investigate model compression and quantisation for resource-limited devices.
- Explore Explainable AI techniques for understanding model predictions.
- Conduct a controlled comparison between original-image and lesion-focused classification.
- Validate the pipeline using clinically verified data.

## 11. Technologies Used

- **Programming:** Python
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV, scikit-image
- **Numerical Computing:** NumPy
- **Data Analysis:** Pandas
- **Machine Learning:** scikit-learn
- **Environment:** Google Colab / Jupyter Notebook

## 12. Disclaimer

This project is an academic research internship project involving deep learning-based skin-lesion segmentation and classification. It is intended for research and educational purposes only. The model predictions are not clinically validated and must not be used as a substitute for professional medical diagnosis.

## 13. Acknowledgements

I would like to express my gratitude to **Dr. Gautam Kumar**, Assistant Professor, Department of Computer Science and Engineering, NIT Delhi, for his guidance and support during this research internship.

I also acknowledge NIT Delhi and Manipal University Jaipur for providing the academic environment and opportunity to undertake this project.

---

**Project Title:** Skin Lesion Segmentation and Classification Using Deep Learning  
**Research Internship:** National Institute of Technology Delhi  
**Duration:** June 2026
