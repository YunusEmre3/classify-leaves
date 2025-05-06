
# Classify Leaves

This project is a deep learning pipeline for classifying leaf images using a fine-tuned **MobileNetV2** architecture with heavy data augmentation.

The project includes:

- Data preparation and augmentation
- Model training with fine-tuning
- Inference and validation
- Pretrained model and label encoder

## 📊 Dataset

The dataset used is the **Classify Leaves** dataset from Kaggle:

[➡️ Classify Leaves - Kaggle Dataset](https://www.kaggle.com/competitions/classify-leaves/data)

The data includes labeled images of different leaf species.  
The `training.ipynb` and `inference.ipynb` notebooks assume the same folder structure as provided by Kaggle.

## 🏗️ Project Structure

```plaintext
classify-leaves/
├── notebooks/
│   ├── training.ipynb
│   └── inference.ipynb
├── models/
│   ├── mobilenetv2_finetuned_xaugmented.keras
│   └── label_encoder.pkl
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Key libraries:
- TensorFlow 2.x
- scikit-learn
- pandas

## 🏋️ Training

The model can be trained using the `training.py` notebook.

Key steps:
- Data loading and preprocessing
- Data augmentation
- MobileNetV2 fine-tuning
- Model checkpointing and early stopping

Output:
- Trained model saved as `mobilenetv2_finetuned_xaugmented.keras`
- Label encoder saved as `label_encoder.pkl`

## 🔎 Inference / Evaluation

To evaluate the trained model, use the `inference.py` notebook.

The notebook:
- Loads the pretrained model and label encoder
- Prepares the validation set
- Evaluates model accuracy

## 📈 Model Performance

Final validation accuracy achieved: **91.99%**

## 📄 License

This project is released under the MIT License.

## 🙌 Acknowledgements

Thanks to Kaggle for providing the dataset and community support.
