import os
import json
import coremltools as ct
import keras
import tensorflow as tf

# ========= CONFIG =========
MODEL_PATH = "dbd_perks_model_finetuned.keras"   # your tuned model
COREML_PATH = "DBDPerkClassifier_Tuned.mlpackage"  # note: .mlpackage (not .mlmodel)
LABEL_JSON = "dbd_perk_labels.json"
IMG_SIZE = (128, 128)

# ========= LOAD MODEL =========
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

print(f"📦 Loading model from {MODEL_PATH} …")
model = keras.models.load_model(MODEL_PATH)

# ========= PATCH FOR TENSORFLOW 2.16+ =========
if not hasattr(tf.keras.Model, "_get_save_spec"):
    def _get_save_spec(self, dynamic_batch=True):
        return None
    tf.keras.Model._get_save_spec = _get_save_spec

# ========= CONVERT TO COREML =========
print("🍏 Converting to CoreML (ML Program format)…")

if os.path.exists(LABEL_JSON):
    with open(LABEL_JSON, "r") as f:
        label_map = json.load(f)
    labels = list(label_map.keys())
else:
    labels = []
    print("⚠️ No label file found. Using numeric class indices.")

print("🍏 Converting to CoreML (with classifier head)…")

classifier_config = ct.ClassifierConfig(class_labels=labels) if labels else None

mlmodel = ct.convert(
    model,
    source="tensorflow",
    convert_to="mlprogram",
    classifier_config=classifier_config,
    inputs=[
        ct.ImageType(
            name="input_1",
            shape=(1, IMG_SIZE[0], IMG_SIZE[1], 3),
            scale=1/255.0,
        )
    ],
)

# ========= ADD METADATA =========
if os.path.exists(LABEL_JSON):
    with open(LABEL_JSON, "r") as f:
        label_map_json = json.load(f)
else:
    label_map_json = {}

mlmodel.short_description = "Fine-tuned Dead by Daylight Perk Classifier"
mlmodel.author = "Annur Alif Ramadhoni"
mlmodel.license = "MIT"
mlmodel.user_defined_metadata["labels_json"] = json.dumps(label_map_json)

# ========= SAVE =========
mlmodel.save(COREML_PATH)
print(f"✅ CoreML model saved as: {COREML_PATH}")
