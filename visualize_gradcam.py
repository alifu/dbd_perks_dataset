import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from rembg import remove

# ====== CONFIG ======
MODEL_PATH = "dbd_perks_model_finetuned.keras"
LABEL_JSON = "dbd_perk_labels.json"
IMG_SIZE = (128, 128)
TRAIN_DIR = "dbd_perks_dataset/augmented"
PROCESSED_DIR = "processed"

# ====== BACKGROUND REMOVAL ======
def preprocess_perk_image(path, target_size=(128, 128)):
    """Crop out the purple background and resize to target_size."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Purple hue range (tuned for DBD perks)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Invert mask to keep white icon only
    mask_inv = cv2.bitwise_not(mask)
    icon = cv2.bitwise_and(img, img, mask=mask_inv)

    # Find bounding box of visible (non-background) region
    coords = cv2.findNonZero(mask_inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = icon[y:y+h, x:x+w]
    else:
        cropped = img  # fallback if no purple background found

    cropped = cv2.resize(cropped, target_size)
    return Image.fromarray(cropped)

# ====== HELPER FUNCTIONS ======
def load_labels(train_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=1)
    return list(gen.class_indices.keys())

def preprocess_image(img_path, target_size):
    """Loads and preprocesses an image (with background removal)."""
    filename = os.path.basename(img_path)
    processed_path = os.path.join(PROCESSED_DIR, filename)
    img = preprocess_perk_image(img_path, target_size=target_size)
    img_no_bg = remove_background(img, save_path=processed_path)
    x = keras_image.img_to_array(img_no_bg)
    x = np.expand_dims(x, axis=0) / 255.0
    return img_no_bg, x

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

# ====== MAIN ======
if len(sys.argv) < 2:
    print("Usage: python visualize_gradcam.py <path_to_image.png>")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"❌ Image not found: {img_path}")
    sys.exit(1)

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

print(f"📦 Loading model from {MODEL_PATH} …")
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
labels = load_labels(TRAIN_DIR)

# Preprocess image
orig_img, x = preprocess_image(img_path, target_size=IMG_SIZE)

# ====== PREDICT & SHOW TOP-5 ======
pred = model.predict(x)[0]

# Get top 5 predictions
top5_idx = np.argsort(pred)[::-1][:10]
top5_labels = [labels[i] for i in top5_idx]
top5_scores = [pred[i] for i in top5_idx]

print("\n🎯 Top 5 Predictions:")
for i, (label, score) in enumerate(zip(top5_labels, top5_scores), start=1):
    print(f" {i}. {label:30s} {score*100:.2f}%")

# Pick the best one for GradCAM visualization
pred_idx = top5_idx[0]
pred_label = top5_labels[0]
print(f"\n🔥 Using top prediction for Grad-CAM: {pred_label}")

# ====== GRAD-CAM ======
# Find last conv layer
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name is None:
    raise ValueError("❌ Could not find a Conv2D layer for Grad-CAM")

print(f"🧠 Using last conv layer: {last_conv_layer_name}")

grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(x)
    loss = predictions[:, pred_idx]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

# Normalize heatmap
heatmap = np.maximum(heatmap, 0)
max_val = heatmap.max()
if max_val != 0:
    heatmap /= max_val

# Resize heatmap to match the original processed image
heatmap = np.uint8(255 * heatmap)
heatmap = np.expand_dims(heatmap, axis=2)
heatmap = tf.image.resize(heatmap, (orig_img.size[1], orig_img.size[0])).numpy().astype("uint8")
heatmap = np.squeeze(heatmap)

# Create color overlay
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap_color * 0.4 + np.array(orig_img)

# ====== DISPLAY RESULTS ======
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original (Processed)")
plt.imshow(orig_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM Overlay")
plt.imshow(superimposed_img.astype("uint8"))
plt.axis("off")

plt.tight_layout()
plt.show()

# ====== SAVE RESULT ======
os.makedirs(PROCESSED_DIR, exist_ok=True)
filename = os.path.basename(img_path)
save_path = os.path.join(PROCESSED_DIR, os.path.splitext(filename)[0] + "_gradcam.png")
cv2.imwrite(save_path, superimposed_img)
print(f"✅ Saved Grad-CAM result to {save_path}")
