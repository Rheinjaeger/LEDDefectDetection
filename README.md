# LED Defect Detection
This serves as the project repository for ECE Machine Learning.

This project implements a binary classification system for detecting defects in LED packaging using deep learning and supervised contrastive learning. The model distinguishes between normal and defective LED units using images collected from two industrial datasets (7020 and Q60B). For information security reasons, the dataset is not included in this public repository as it is from an actual industrial source.

## Overview

- **Backbone Model**: ResNet-50.
- **Feature Learning**: Supervised Contrastive Learning (SupConLoss) with augmentations.
- **Classification**: A linear classifier trained to distinguish normal vs. defect.
- **Datasets**: 7020 and Q60B datasets with normal and defective LED samples.
- **Sampling**: Weighted sampling to address class imbalance.
- **Evaluation Metrics**: Accuracy, F1-score, Precision, Recall, and Confusion Matrix.

The training pipeline is divided into two stages:
1. **Stage 1**: Joint training with SupConLoss and CrossEntropyLoss.
2. **Stage 2**: Fine-tuning with only CrossEntropyLoss.

## Project Structure

- `resnet.py`: Contains the modified ResNet-50 encoder, supervised contrastive head, and binary classifier.
- `losses.py`: Implements Supervised Contrastive Loss (`SupConLoss`) as proposed by Khosla et al.
- `dataset_SupContrast.py`: Loads and processes the 7020 and Q60B datasets, applies augmentations, and builds training/testing sets.
- `train.py`: Runs the full training and evaluation pipeline.

## How to Run
- Prepare the required environment (IDE and correct interpreter with necessary libraries and frameworks).
- Run `train.py` directly.
