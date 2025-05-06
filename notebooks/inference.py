from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import pickle

# Dataset
DATA_PATH="/kaggle/input/classify-leaves"

base_path = os.getenv("DATA_PATH", "data/classify-leaves")
df = pd.read_csv(os.path.join(base_path, "train.csv"))
df["image_path"] = df["image"].apply(lambda x: os.path.join(base_path, "images", x.replace("images/", "")))

ENCODER_PATH="/kaggle/input/mobilenetv2-finetuned-xaugmented-final/label_encoder.pkl \"

# Label Encoder
encoder_path = os.getenv("ENCODER_PATH", "models/label_encoder.pkl")

if os.path.exists(encoder_path):
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
else:
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

if "label_encoded" not in df.columns:
    df["label_encoded"] = label_encoder.transform(df["label"])

# Validation set
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label_encoded"], random_state=42)

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label_encoded",
    target_size=(224, 224),
    batch_size=64,
    class_mode="raw",
    shuffle=False
)

MODEL_PATH="/kaggle/input/mobilenetv2-finetuned-xaugmented-final/mobilenetv2_finetuned_xaugmented.keras \"

model_path = os.getenv("MODEL_PATH", "models/mobilenetv2_finetuned_xaugmented.keras")
# Model load
model = load_model(model_path)

# Evaluate
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
