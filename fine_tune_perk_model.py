import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

MODEL_PATH = "dbd_perks_model.keras"
DATA_DIR = "dbd_perks_dataset/augmented"
FINE_TUNED_MODEL_PATH = "dbd_perks_model_finetuned.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5  # can increase after verifying

# ==========================
# LOAD EXISTING MODEL
# ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

print(f"📦 Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# ==========================
# DETECT BASE OR CUSTOM MODEL
# ==========================
base_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model:
    print(f"✅ Found base model: {base_model.name}")
    unfreeze_from = max(len(base_model.layers) - 60, 0)
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
else:
    print("ℹ️ No nested base model found. Assuming custom CNN.")
    # Unfreeze last 1/3 of layers
    unfreeze_from = int(len(model.layers) * 0.66)
    for layer in model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in model.layers[unfreeze_from:]:
        layer.trainable = True

# ==========================
# DATASET
# ==========================
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
)

# Compute class weights
class_labels = list(train_gen.class_indices.keys())
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_labels)),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))

# ==========================
# RECOMPILE
# ==========================
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# FINE-TUNE TRAINING
# ==========================
print("🚀 Fine-tuning model...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
)

# ==========================
# SAVE MODEL
# ==========================
model.save(FINE_TUNED_MODEL_PATH)
print(f"✅ Fine-tuned model saved as {FINE_TUNED_MODEL_PATH}")
