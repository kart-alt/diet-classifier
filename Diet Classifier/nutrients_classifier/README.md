# 🍽️ AI Food Nutrients Classifier

A complete End-to-End deep learning pipeline that detects food from an uploaded image, estimates the portion size, and calculates the total nutritional breakdown in a production-ready Web App.

## 🚀 Key Features

- **Trained entirely from scratch**: No pre-trained ImageNet weights. The model learns visual features purely from a massive unified dataset of food images.
- **Portion Size Estimator**: Predicts the weight logically using a secondary regression network.
- **Nutrition Context Engine**: Utilizes explicit hashing (O(1) mapping) into the USDA Nutritional Database for instantaneous macronutrient calculations.
- **Bilingual Support**: Fully toggleable between English and Tamil for users in diverse demographics.
- **Export to PDF**: Generate structured `Nutrition Facts` reports on the fly. 

## 🏗️ Architecture

```text
UPLOAD ──► PREPROCESS ──► CNN FEATURE EXTRACTOR (Custom 5-Block w/ Residuals + SE)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     CLASSIFICATION HEAD                PORTION REGRESSOR
     (Softmax - 101/256 classes)        (Linear Grams Est.)
                 │                               │
                 └──────────────►◄───────────────┘
                                 │
                          NUTRITION ENGINE (USDA Mapping)
                                 │
                         GRADIO WEB INTERFACE
                          (Charts, PDF, Facts)
```

## 🧠 The Custom CNN (Train from Scratch)
Because this project strictly forbids ImageNet transfer learning, we designed a robust, modern deep network to ensure convergence:
1. **Residual Connections** to prevent vanishing gradients during early epochs.
2. **Squeeze-and-Excitation (SE) Blocks** dynamically recalibrate channel-wise feature responses just before classification.
3. **Mixed Precision Training** (`mixed_float16`) reduces GPU VRAM consumption by 50% allowing heavy augmentations (MixUp, Cutout) to run rapidly without throwing OOM errors.
4. **Cosine Annealing** learning rate scheduler pushes the model out of local minima.

## ⚙️ Setup and Installation

Requirements: Python 3.10+, and a CUDA-capable GPU is highly recommended.

1. Clone or navigate to the project directory:
   ```bash
   cd nutrients_classifier
   ```
2. Execute the Setup Script (Linux/Mac):
   ```bash
   bash setup.sh
   ```
   *(On Windows, run setup steps line-by-line or use Git Bash)*

### Manual Pipeline Execution

To individually trigger pipeline components:
1. **Download Data:** `python src/download_datasets.py`
2. **Preprocess (create TFRecords & Mapping):** `python src/preprocess.py`
3. **Train the Models:** `python src/train_model.py`
4. **Run Inference on a single image:** `python src/predict.py data/raw/pizza.jpg`
5. **Launch the UI UI:** `python app.py`

## 📊 Dataset Credits
- **Food-101**: Bossard, Lukas, et al. "Food-101–mining discriminative components with random forests." (ECCV 2014)
- **UEC Food-256**: Kawano, Yoshiyuki, and Keiji Yanai. "Automatic expansion of a food image dataset leveraging existing categories with domain adaptation."
- **Nutrition5k** (Portion Regression reference dataset) by Google Research.
- **USDA Nutritional Database**: Open USDA open data catalog.

*License: Please observe the specific data licenses for Food-101 and UEC-256 related to non-commercial / research usage.*
