# Classify Leaves

This project contains a deep learning model built for the Kaggle "Classify Leaves" challenge.

## 🏆 Validation Accuracy

Achieved **91.99%** validation accuracy with MobileNetV2 using heavy data augmentation and fine-tuning.

## 📁 Project Structure

- `notebooks/training.py` → Model training script with data augmentation and fine-tuning.
- `notebooks/inference.py` → Script to evaluate the trained model and test accuracy.
- `models/mobilenetv2_finetuned_xaugmented.keras` → Trained model file.
- `models/label_encoder.pkl` → Label encoder used during training and inference.
- `data/` → Placeholder for dataset-related files (empty for now).

## 🚀 How to Use

```bash
pip install -r requirements.txt
