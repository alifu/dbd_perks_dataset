import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# ==========================
# CONFIGURATION
# ==========================
DATA_DIR = "dbd_perks_dataset/augmented"
METADATA_PATH = "dbd_perks_dataset/metadata.json"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = "dbd_perks_model.keras"
LABEL_JSON = "dbd_perk_labels.json"

# ==========================
# LOAD METADATA
# ==========================
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"❌ metadata.json not found at {METADATA_PATH}")

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# ==========================
# DATA GENERATORS
# ==========================
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

# ==========================
# MODEL DEFINITION
# ==========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

print("🚀 Starting initial training...")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=1)

# Fine-tuning last layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
print("🔧 Fine-tuning model...")
model.fit(train_gen, validation_data=val_gen, epochs=5, verbose=1)

# ==========================
# SAVE MODEL
# ==========================
model.save(MODEL_NAME)
print(f"✅ Model saved as {MODEL_NAME}")

# ==========================
# LABEL MAP
# ==========================
labels = list(train_gen.class_indices.keys())
label_map = {}

for label in labels:
    clean_key = label.replace("File:IconPerks_", "")
    meta = metadata.get(label) or metadata.get(clean_key)
    if meta:
        label_map[label] = {
            "name": meta.get("name", clean_key),
            "description": meta.get("description", "")
        }
    else:
        label_map[label] = {"name": clean_key, "description": ""}

with open(LABEL_JSON, "w") as f:
    json.dump(label_map, f, indent=2)

print(f"✅ Label map saved → {LABEL_JSON}")
