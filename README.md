# DBD Perks Dataset & Classifier  
*A machine‑learning pipeline for classifying perks in Dead by Daylight*

This repository contains a full workflow to:

- scrape perk icons & descriptions  
- build a dataset (raw + augmented images)  
- train a CNN classifier  
- convert the model to CoreML for iOS/macOS use  
- run quick tests for inference  

> **Note:** This project is for educational purposes only.

---

## 📁 Repository Structure

```
dbd_perks_dataset/
│
├── raw/ ← Original perk icon images (1 sample per perk)
│ ├── File:IconPerks_aceInTheHole.png
│ ├── File:IconPerks_barbecueAndChilli.png
│ └── …
│
├── augmented/ ← Augmented image folders for each perk
│ ├── File:IconPerks_aceInTheHole/
│ │ ├── File:IconPerks_aceInTheHole_0.png
│ │ └── …
│ ├── File:IconPerks_barbecueAndChilli/
│ │ ├── File:IconPerks_barbecueAndChilli_0.png
│ │ └── …
│
├── metadata.json ← JSON mapping perk keys → name, description, page URL
├── dbd_perk_labels.json ← JSON mapping classifier labels → display name + description
│
├── train_dbd_perks_model.py ← Script to train the Keras model
├── convert_to_core_ml_model.py ← Script to convert trained model to CoreML
├── quick_test_model.py ← Script to quickly test inference on a single image
│
└── build_dbd_perk_dataset_api.py ← Web‑scraper to build dataset from the wiki
```

---

## 🛠 Workflow

### 1. Build the dataset  
Use the scraper script (e.g. `build_dbd_perk_dataset_api.py`) to download perk icon images and descriptions from the Dead by Daylight Fandom wiki, then run the augmentation script to generate additional training images (~50 per perk).

### 2. Train the model  
Use `train_dbd_perks_model.py`. This script:
- loads the augmented images  
- splits into training / validation sets  
- uses a pre‑trained model (e.g. MobileNetV2) for transfer‑learning  
- fine‑tunes the model  
- saves the model (e.g. `dbd_perks_model.keras`)  
- produces a label‑map JSON (`dbd_perk_labels.json`)

### 3. Convert to CoreML  
Use `convert_to_core_ml_model.py`. This script:
- loads the saved Keras model  
- optionally patches compatibility issues  
- converts to a CoreML format (`.mlpackage` for newer iOS/macOS or `.mlmodel` for older)  
- embeds metadata (labels + descriptions)  
- saves the output (e.g. `DBDPerkClassifier.mlpackage`)

### 4. Quick test / Inference  
Use `quick_test_model.py` to run inference on a sample perk icon from `raw/`.  
It loads the model, finds a sample image, preprocesses it (resize + normalize), runs prediction, and prints the top result with confidence + optional description.

---

## ⚠️ Compatibility & Notes

- *TensorFlow / Keras versions:*  
  Due to compatibility issues, it’s recommended to use TensorFlow **2.12.0** with its bundled Keras.  
  Newer TensorFlow versions (2.15+) may require manual patches for CoreML conversion.  
- *CoreML format:*  
  - For iOS 15 / macOS 12 and newer: use `.mlpackage` (ML Program format)  
  - For older systems: specify `convert_to="neuralnetwork"` and save as `.mlmodel`.  
- *Dataset details:*  
  - One raw icon image per perk  
  - ~50 augmented images per perk folder  
  - Metadata includes perk name, image file reference, description, and wiki page URL  
- This project is **for learning purposes only**. Please respect the rights and usage policies of the original game and wiki content.

---

## 📋 Example Usage (Terminal)

```bash
# 1) Train the model
python3 train_dbd_perks_model.py

# 2) Convert the model to CoreML
python3 convert_to_core_ml_model.py

# 3) Run a quick test
python3 quick_test_model.py
```

---

## 🧠 Model Conversion & Visualization

This section explains how to convert your fine-tuned Dead by Daylight perk classifier to CoreML for iOS apps, and how to visualize Grad-CAM heatmaps to understand the model’s predictions.

### 🍏 Convert to CoreML (`convert_tuned_to_coreml.py`)

After you have fine-tuned your model (tuned_dbd_perks_model.keras), you can convert it into a CoreML .mlpackage model for integration into your iOS app.

Usage
```bash
python convert_tuned_to_coreml.py
```

Script Overview

- Loads your fine-tuned .keras model.
- Converts it to a .mlpackage (the modern CoreML format).
- Embeds metadata such as author, license, and JSON label mapping.
- Saves the result as DBDPerkClassifier.mlpackage.

Output
```bash
✅ CoreML model saved as DBDPerkClassifier.mlpackage
```


You can now import this file into Xcode and use it via VNCoreMLModel or Swift’s CoreML framework for on-device inference.

## 🔥 Grad-CAM Visualization (visualize_gradcam.py)

Grad-CAM helps you see which regions of a perk image the model focuses on during classification — perfect for debugging or improving dataset accuracy.

Usage
```bash
python visualize_gradcam.py path/to/perk_image.png
```

Example
```bash
python visualize_gradcam.py samples/a_nurse_calling.png
```

What it does
- Loads your trained model (dbd_perks_model.keras or fine-tuned one).
- Preprocesses the input image and predicts its class.
- Displays the top 5 most likely perks with confidence percentages.
- Generates a Grad-CAM overlay to highlight image regions influencing the decision.

Output Example
```bash
🎯 Top 5 Predictions:
 1. A Nurse's Calling             92.34%
 2. Dark Theory                   4.28%
 3. Kindred                       1.65%
 4. Empathy                       0.87%
 5. Bond                           0.42%

🔥 Using top prediction for Grad-CAM: A Nurse's Calling
🧠 Using last conv layer: conv2d_94
✅ Saved Grad-CAM result to processed/a_nurse_calling_gradcam.png
```

It also displays side-by-side:

- The original perk image
- The Grad-CAM overlay showing attention regions
(red = high importance, blue = low importance)

Example Output

The script automatically saves:
```bash
processed/<filename>_gradcam.png
```

## ⚙️ Requirements

Ensure you have these Python dependencies installed:
```bash
pip install tensorflow coremltools opencv-python matplotlib pillow numpy
```

Optional (for fine-tuning and dataset management):
```bash
pip install scikit-learn
```

---

## 🙋 Contact & Contributions

This project was created by **Annur Alif Ramadhoni**.
Feel free to open issues or submit pull requests for improvements (e.g., support for new perks, mobile‑app integration, further model optimization).

---

## 🎯 License & Attribution

The model, scripts, and dataset in this repository are provided with MIT License (see LICENSE file).

Icon images and descriptions are sourced from the [Dead by Daylight Wiki (Fandom)](https://deadbydaylight.fandom.com).
Please refer to the original wiki for usage rights and attribution.