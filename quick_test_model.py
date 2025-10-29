import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from rembg import remove

# ==========================
# CONFIG
# ==========================
IMG_SIZE = (128, 128)
MODEL_PATH = "dbd_perks_model.keras"
TRAIN_DIR = "dbd_perks_dataset/augmented"
LABEL_JSON = "dbd_perk_labels.json"
PROCESSED_DIR = "processed"

# ==========================
# LOAD MODEL
# ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# ==========================
# LOAD LABELS
# ==========================
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=1)
labels = list(train_gen.class_indices.keys())

# Load optional label metadata
label_info = {}
if os.path.exists(LABEL_JSON):
    with open(LABEL_JSON, "r") as f:
        label_info = json.load(f)

# ==========================
# FUNCTIONS
# ==========================
def remove_background(img: Image.Image, save_path: str):
    """
    Remove background using rembg and save the processed image.
    """
    output = remove(img)
    output_rgb = output.convert("RGB")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_rgb.save(save_path)
    print(f"✅ Saved processed image: {save_path}")
    return output_rgb

def predict_image(img: Image.Image):
    """
    Resize, normalize, and predict top 3 classes.
    """
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)
    
    pred = model.predict(x)[0]
    top_indices = np.argsort(pred)[::-1][:3]

    print("\n🎯 Top Predictions:")
    for i in top_indices:
        class_name = labels[i]
        confidence = pred[i] * 100
        info = label_info.get(class_name, {})
        print(f"- {info.get('name', class_name)} ({confidence:.2f}%)")
        if info.get("description"):
            print(f"  📖 {info['description'][:120]}{'...' if len(info['description']) > 120 else ''}")

# ==========================
# MAIN
# ==========================
input_path = input("Enter path to image to predict: ").strip()
if not os.path.exists(input_path):
    print("❌ File not found!")
    exit()

filename = os.path.basename(input_path)
processed_path = os.path.join(PROCESSED_DIR, filename)

# Remove background and save
img = Image.open(input_path)
img_no_bg = remove_background(img, save_path=processed_path)

# Predict
predict_image(img_no_bg)
