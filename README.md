# Transfer Learning for Object Recognition (Caltech-101)

## Overview

This project explores the effectiveness of **transfer learning** for multi-class image classification using pretrained convolutional neural networks. Three architectures — **MobileNetV2**, **EfficientNetB0**, and **ResNet18** — were trained and evaluated on the **Caltech-101 dataset**.

The goal was to compare:

- Feature Extraction (Frozen Backbone)
- Fine-Tuning (Unfrozen Backbone)
- Validation vs Test Performance
- Model Efficiency vs Accuracy

---

## Dataset

**Caltech-101**

- 101 object categories (+ background class)
- ~9,000 total images
- ~40–800 images per category
- Images resized to **224 × 224**
- Data augmentation applied:
  - Random horizontal flip
  - Random rotation
  - Random zoom

Each architecture used its own dataset split strategy during experimentation.

---

## Models Implemented

### 1️⃣ MobileNetV2

Lightweight architecture using depthwise separable convolutions, designed for efficiency.

**Training Strategy**
- Phase 1: Backbone frozen (feature extraction)
- Phase 2: Backbone unfrozen (fine-tuning with lower learning rate)

**Performance**
- Frozen Validation Accuracy: **83.9%**
- Fine-Tuned Validation Accuracy: **87.7%**
- Test Accuracy: **82.13%**

MobileNetV2 delivered competitive accuracy with significantly lower computational cost.

---

### 2️⃣ EfficientNetB0

Architecture based on compound scaling (depth, width, resolution).

**Training Strategy**
- Feature extraction phase
- Fine-tuning phase with reduced learning rate

**Performance**
- Frozen Validation Accuracy: **96.0%**
- Fine-Tuned Validation Accuracy: **91.1%**
- Test Accuracy: **97.02%** (Best performing model)

EfficientNetB0 achieved the highest test accuracy and demonstrated strong generalization capability.

---

### 3️⃣ ResNet18

Residual network using skip connections for stable gradient flow.

**Training Strategy**
- Frozen backbone training
- Fine-tuning with smaller learning rate

**Performance**
- Frozen Validation Accuracy: **90.75%**
- Fine-Tuned Validation Accuracy: **94.43%**
- Test Accuracy: (Add if available)

ResNet18 showed stable convergence and strong validation performance after fine-tuning.

---

## Training Details

- Framework: TensorFlow / Keras
- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Cross-Entropy**
- Input Size: 224 × 224
- Batch Size: 32
- Fine-Tuning Learning Rate: 1e-5
- EarlyStopping used to prevent overfitting
- ModelCheckpoint used to save best-performing weights

---

## Key Observations

- Fine-tuning consistently improved model performance across all architectures.
- EfficientNetB0 achieved the highest overall accuracy (97.02%).
- ResNet18 demonstrated strong stability and reliable convergence.
- MobileNetV2 provided a strong efficiency-to-performance tradeoff.
- Updating convolutional layers during fine-tuning significantly improved classification compared to feature extraction alone.

---

## Confusion Matrix

The confusion matrix of the best-performing model (EfficientNetB0) is included in this repository. It shows strong diagonal dominance, indicating accurate class predictions across most categories.

---

## Repository Structure
├── mobilenet.ipynb
├── efficientnet.ipynb
├── resnet.ipynb
├── final_paper.pdf
└── README.md


---

## Conclusion

This project demonstrates the effectiveness of transfer learning for moderate-scale image classification tasks. Fine-tuning pretrained convolutional backbones significantly improves generalization compared to using frozen feature extractors alone.

EfficientNetB0 achieved the strongest performance, while MobileNetV2 offered competitive accuracy with reduced computational complexity.

---

## Future Improvements

- Hyperparameter optimization
- Learning rate scheduling
- Class imbalance handling
- Top-5 accuracy evaluation
- Deployment optimization for edge devices


