from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import pickle

# 2. Load dataset CSV and prepare image paths
# If running on Kaggle, use Kaggle input path; otherwise, use local 'data/classify-leaves'
if os.path.exists("/kaggle/input/classify-leaves"):
    base_path = "/kaggle/input/classify-leaves"
else:
    base_path = "data/classify-leaves"  # Make sure to download and place the dataset here locally

df = pd.read_csv(os.path.join(base_path, "train.csv"))
df["image_path"] = df["image"].apply(lambda x: os.path.join(base_path, "images", x.replace("images/", "")))

df = pd.read_csv(os.path.join(base_path, "train.csv"))
df["image_path"] = df["image"].apply(lambda x: os.path.join(base_path, "images", x.replace("images/", "")))

# 3. Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# 4. Train-validation split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label_encoded"], random_state=42)

# 5. Data generators with heavy augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label_encoded",
    target_size=(224, 224),
    batch_size=64,
    class_mode="raw"
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label_encoded",
    target_size=(224, 224),
    batch_size=64,
    class_mode="raw"
)

# 6. Model setup
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = True  # Fine-tuning all layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(df["label_encoded"].nunique(), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# 7. Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 8. Callbacks
checkpoint = ModelCheckpoint(
    "mobilenetv2_finetuned_xaugmented.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-7
)

# 9. Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=200,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)
