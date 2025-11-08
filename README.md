# 🩺 Breast Ultrasound Cancer Detection using Segmentation-Guided Deep Learning

This repository contains the implementation of a **deep learning pipeline** for **breast cancer detection from ultrasound images**, integrating **lesion segmentation (U-Net++)** and **image classification (EfficientNetB0)**.

---

## 🌟 Overview

Breast ultrasound is a non-invasive imaging technique commonly used for early cancer screening.  
However, interpreting ultrasound scans can be challenging and depends on radiologist expertise.

This project proposes a **segmentation-guided classification pipeline** that combines lesion localization and diagnosis prediction — aiming to improve model robustness and interpretability.

---

## ⚙️ Architecture

### 1. **Lesion Segmentation (U-Net++)**
- Input: Grayscale breast ultrasound images  
- Output: Binary lesion mask  
- Loss: Combined **Binary Cross-Entropy + Dice Loss**

### 2. **Segmentation-Guided Classification (EfficientNetB0)**
- Input: 3-channel fusion of  
  - Original grayscale image  
  - Contrast-enhanced lesion image  
  - Binary segmentation mask  
- Output: Benign vs. Malignant classification  
- Trained using **transfer learning** with ImageNet weights

---

## 🧩 Pipeline Diagram

```text
Ultrasound Image → U-Net++ Segmentation → Lesion Mask
      ↓
Contrast Enhancement + Mask Fusion → 3-Channel Image
      ↓
EfficientNetB0 Classifier → Benign / Malignant
# 🩺 Breast Ultrasound Cancer Detection using Segmentation-Guided Deep Learning

This repository contains the implementation of a **deep learning pipeline** for **breast cancer detection from ultrasound images**, integrating **lesion segmentation (U-Net++)** and **image classification (EfficientNetB0)**.

---

## 🌟 Overview

Breast ultrasound is a non-invasive imaging technique commonly used for early cancer screening.  
However, interpreting ultrasound scans can be challenging and depends on radiologist expertise.

This project proposes a **segmentation-guided classification pipeline** that combines lesion localization and diagnosis prediction — aiming to improve model robustness and interpretability.

---

## ⚙️ Architecture

### 1. **Lesion Segmentation (U-Net++)**
- Input: Grayscale breast ultrasound images  
- Output: Binary lesion mask  
- Loss: Combined **Binary Cross-Entropy + Dice Loss**

### 2. **Segmentation-Guided Classification (EfficientNetB0)**
- Input: 3-channel fusion of  
  - Original grayscale image  
  - Contrast-enhanced lesion image  
  - Binary segmentation mask  
- Output: Benign vs. Malignant classification  
- Trained using **transfer learning** with ImageNet weights

---

## 🧩 Pipeline Diagram

```text
Ultrasound Image → U-Net++ Segmentation → Lesion Mask
      ↓
Contrast Enhancement + Mask Fusion → 3-Channel Image
      ↓
EfficientNetB0 Classifier → Benign / Malignant
📊 Results (BUSI Dataset)
Model	Accuracy	Malignant Recall	AUC	Precision	Dice (Segmentation)
ResNet50 (baseline)	81%	55%	0.83	0.78	-
EfficientNetB0 (no segmentation)	84%	72%	0.94	0.86	-
Proposed U-Net++ + EfficientNetB0	84%	94%	0.94	0.91	0.89
🔍 Explainability (Grad-CAM++)

Model predictions were visualized using Grad-CAM++, highlighting diagnostically relevant regions in malignant lesions.

Original	Tumor Crop	Grad-CAM++

	
🚀 Installation
# Clone the repository
git clone https://github.com/yourusername/BreastCancerSegClass.git](https://github.com/mohammadreza-tabatabaei/Breast-Ultrasound-Cancer-Detection-using-Segmentation-Guided-Deep-Learning
cd BreastCancerSegClass

# Install dependencies
pip install -r requirements.txt

🧠 Training
1. Train U-Net++ for segmentation
python train_unetpp.py

2. Train EfficientNetB0 with segmentation guidance
python train_seg_guided_classifier.py

3. Generate Grad-CAM++ visualizations
python gradcam_visualization.py

🧪 Dataset

BUSI Dataset (Breast Ultrasound Images Dataset)
Al-Dhabyani et al., Data in Brief, 2020

Includes 780 images labeled as normal, benign, or malignant with pixel-level masks.

⚠️ Dataset must be downloaded manually and placed under:

/Dataset_BUSI_with_GT/
├── benign/
├── malignant/

🧾 Requirements

Python 3.8+

TensorFlow 2.x

Keras

OpenCV

Albumentations

NumPy, Matplotlib, scikit-learn

Install all dependencies:

pip install tensorflow keras opencv-python albumentations scikit-learn matplotlib tqdm

💡 Future Work

Incorporate 3D and temporal ultrasound data

Include clinical metadata (age, BI-RADS, lesion history)

Explore end-to-end multi-task learning

Enhance interpretability using SHAP or LRP

🧑‍💻 Author

Mohammadreza Tabatabaei
MSc Dissertation – Biomedical Image Analysis & Deep Learning
📍 Manchester Metropolitan University
🔗 https://www.linkedin.com/in/mohammadreza-tabatabaei-057510250/

📧 Tabatabaei.mhrz@gmail.com
