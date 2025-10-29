import os
import json
import coremltools as ct
import keras   # ✅ use Keras 3, not tf.keras

MODEL_NAME = "dbd_perks_model.keras"
COREML_NAME = "DBDPerkClassifier.mlpackage"
LABEL_JSON = "dbd_perk_labels.json"
IMG_SIZE = (128, 128)

# ==========================
# LOAD MODEL
# ==========================
if not os.path.exists(MODEL_NAME):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_NAME}")

print("📦 Loading trained model...")
model = keras.models.load_model(MODEL_NAME)

# ==========================
# PATCH FOR COREMLTOOLS
# ==========================
import tensorflow as tf
if not hasattr(tf.keras.Model, "_get_save_spec"):
    def _get_save_spec(self, dynamic_batch=True):
        return None
    tf.keras.Model._get_save_spec = _get_save_spec

# ==========================
# CONVERT TO COREML
# ==========================
print("🍏 Converting to CoreML...")

mlmodel = ct.convert(
    model,
    source="tensorflow",
    inputs=[
        ct.ImageType(
            name="input_1",   # ✅ match your model’s real input name
            shape=(1, IMG_SIZE[0], IMG_SIZE[1], 3),
            scale=1 / 255.0
        )
    ],
)

# ==========================
# ATTACH METADATA
# ==========================
if os.path.exists(LABEL_JSON):
    with open(LABEL_JSON, "r") as f:
        label_map_json = json.load(f)
else:
    label_map_json = {}

mlmodel.short_description = "Dead by Daylight Perk Classifier"
mlmodel.author = "Annur Alif Ramadhoni"
mlmodel.license = "MIT"
mlmodel.user_defined_metadata["labels_json"] = json.dumps(label_map_json)

mlmodel.save(COREML_NAME)
print(f"✅ CoreML model saved as {COREML_NAME}")
