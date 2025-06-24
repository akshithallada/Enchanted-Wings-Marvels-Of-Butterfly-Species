import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------ CONFIG ------------------ #
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "best_model_mobilenet.keras"
ENCODER_PATH = "label_encoder.pkl"
AUTOTUNE = tf.data.AUTOTUNE

base_path = os.path.join(os.getcwd(), "dataset")
train_csv = os.path.join(base_path, "Training_set.csv")
train_img_folder = os.path.join(base_path, "train")

# ------------------ LOAD DATA ------------------ #
train_df = pd.read_csv(train_csv)
train_df['file_path'] = train_df['filename'].apply(lambda x: os.path.join(train_img_folder, x))

label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

# Save the label encoder
joblib.dump(label_encoder, ENCODER_PATH)

train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['encoded_label'], random_state=42)

# ------------------ DATA PIPELINE ------------------ #
def load_image_with_label(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

def build_dataset(df):
    paths = df['file_path'].values
    labels = df['encoded_label'].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image_with_label, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = build_dataset(train_data)
val_ds = build_dataset(val_data)

# ------------------ MODEL SETUP ------------------ #
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(len(label_encoder.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ------------------ TRAINING ------------------ #
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True)

print("\nTraining MobileNetV2 model for 10 epochs...\n")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop, checkpoint], verbose=1)

# ------------------ SAVE MODEL ------------------ #
model.save(MODEL_PATH)
print(f"\n Model saved to {MODEL_PATH}")
print(f"Label encoder saved to {ENCODER_PATH}")
