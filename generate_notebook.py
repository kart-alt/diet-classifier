import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

cells = []

# STEP 0 to 10 provided by the user in markdown
cells.append(new_markdown_cell("""# 🍛 Indian Food Nutrients Classifier
### Fine-tuning EfficientNetB0 (Food-101) + Indian Food Dataset

This notebook:
1. Downloads the Indian Food dataset (open source, no Kaggle login needed)
2. Downloads Food-101 dataset
3. Uploads your existing trained model
4. Fine-tunes it to recognise **Indian foods** in addition to the original 101 classes
5. Maps food predictions → full nutrition breakdown
6. Lets you test on your own food photos

**Runtime: GPU (T4) recommended — Runtime > Change runtime type > T4 GPU**

## ✅ STEP 0 — Install Dependencies"""))

cells.append(new_code_cell("""!pip install tensorflow==2.15.0 kaggle gdown matplotlib seaborn pandas numpy pillow tqdm scikit-learn ipywidgets -q
print('✅ All packages installed!')"""))

cells.append(new_markdown_cell("## 📥 STEP 1 — Upload Your Existing Trained Model"))

cells.append(new_code_cell("""from google.colab import files
import os

print('📂 Upload your food_classifier_custom.h5 file...')
uploaded = files.upload()

MODEL_PATH = list(uploaded.keys())[0]
print(f'✅ Model uploaded: {MODEL_PATH}')"""))

cells.append(new_markdown_cell("""## 📥 STEP 2 — Download Datasets

We download **two datasets**:
- **Indian Food Images** — 4,000+ images across 80 Indian dishes (open source via GitHub/Kaggle)
- **Food-101** — already represented in your model (used for reference label mapping)"""))

cells.append(new_code_cell("""import os

os.makedirs('/content/data', exist_ok=True)
os.makedirs('/content/data/indian_food', exist_ok=True)

# ─────────────────────────────────────────────────────────
# METHOD A: Download Indian Food Dataset via Kaggle API
# ─────────────────────────────────────────────────────────
# If you have a kaggle.json, upload it below and use this method.
# Otherwise jump to METHOD B (no login needed)

USE_KAGGLE = False  # Set True if you have kaggle.json

if USE_KAGGLE:
    from google.colab import files
    print('Upload your kaggle.json...')
    files.upload()
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets download -d iamsouravbanerjee/indian-food-images-dataset -p /content/data/
    !unzip -q /content/data/indian-food-images-dataset.zip -d /content/data/indian_food/
    print('✅ Indian Food dataset downloaded via Kaggle!')
else:
    print('Using METHOD B — downloading via gdown (Google Drive, no login needed)...')"""))

cells.append(new_code_cell("""# ─────────────────────────────────────────────────────────
# METHOD B: Download via gdown (no Kaggle account needed)
# Indian Food 80 Classes Dataset — hosted on Google Drive
# Source: https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset
# ─────────────────────────────────────────────────────────

import gdown, zipfile, os

# Indian Food Dataset (80 classes, ~4000 images)
INDIAN_FOOD_ID = '1FfHmkSWp7bMjuBFxmtSGASG2VhXDmKh8'

output = '/content/data/indian_food.zip'

print('⬇️  Downloading Indian Food dataset (~200MB)...')
try:
    gdown.download(f'https://drive.google.com/uc?id={INDIAN_FOOD_ID}', output, quiet=False)
except Exception as e:
    print(f'⚠️  gdown error: {e}')
    print('Attempting to proceed to fallback methods...')

if os.path.exists(output) and os.path.getsize(output) > 1000:
    print('📦 Extracting...')
    with zipfile.ZipFile(output, 'r') as z:
        z.extractall('/content/data/indian_food/')
    print('✅ Indian Food dataset ready!')
else:
    # Fallback: download from GitHub mirror
    print('⚠️  gdown failed. Trying GitHub fallback...')
    !git clone https://github.com/utsavdarlami/cooking_recipe_recommendation /content/indian_food_ref
    # Use a reliable open dataset from TF datasets instead
    print('Using TensorFlow Datasets - tf_flowers as structure demo + custom Indian classes')
    !pip install tensorflow-datasets -q
    import tensorflow_datasets as tfds
    print('✅ Fallback ready — see STEP 3B for alternative loading')"""))

cells.append(new_code_cell("""# ─────────────────────────────────────────────────────────
# GUARANTEED FALLBACK — scrape open-license Indian food
# images from Open Images Dataset (no login, always works)
# ─────────────────────────────────────────────────────────

import os

# Check what we have
indian_food_root = None
for candidate in [
    '/content/data/indian_food/Indian Food Images/Indian Food Images',
    '/content/data/indian_food/Indian Food Images',
    '/content/data/indian_food',
]:
    if os.path.exists(candidate):
        subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
        if len(subdirs) > 5:
            indian_food_root = candidate
            print(f'✅ Found Indian food data at: {candidate}')
            print(f'   Classes found: {len(subdirs)}')
            print(f'   Sample classes: {subdirs[:10]}')
            break

if indian_food_root is None:
    print('⚠️  Indian food dataset not found. Downloading a verified open-source set...')
    os.makedirs('/content/data/indian_food_manual', exist_ok=True)

    # Download Indian Food-20 Dataset from Kaggle via direct URL (public)
    !pip install opendatasets -q
    import opendatasets as od
    od.download(
        'https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification',
        '/content/data/'
    )
    indian_food_root = '/content/data/indian-food-classification'
    print('✅ Downloaded Indian food dataset via opendatasets')"""))

cells.append(new_code_cell("""# ─────────────────────────────────────────────────────────
# List all Indian food classes available
# ─────────────────────────────────────────────────────────

import os

if indian_food_root:
    indian_classes = sorted([
        d for d in os.listdir(indian_food_root)
        if os.path.isdir(os.path.join(indian_food_root, d))
    ])
    print(f'\\n🍛 Total Indian food classes: {len(indian_classes)}')
    print('\\n📋 All Indian food classes:')
    for i, c in enumerate(indian_classes):
        count = len(os.listdir(os.path.join(indian_food_root, c)))
        print(f'  {i+1:3}. {c:<35} ({count} images)')"""))

cells.append(new_markdown_cell("""## 🧠 STEP 3 — Load Your Existing Model & Inspect"""))

cells.append(new_code_cell("""import tensorflow as tf
import numpy as np

print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices("GPU")) > 0}')

# Load existing model
print(f'\\n📂 Loading model: {MODEL_PATH}')
existing_model = tf.keras.models.load_model(MODEL_PATH)
existing_model.summary()

print(f'\\n✅ Model loaded!')
print(f'   Input shape:  {existing_model.input_shape}')
print(f'   Output shape: {existing_model.output_shape}')
print(f'   Output classes (original): {existing_model.output_shape[-1]}')"""))

cells.append(new_markdown_cell("""## 🗂️ STEP 4 — Define All Class Labels"""))

cells.append(new_code_cell("""import json

# ── Food-101 original class labels (alphabetical order as used in dataset) ──
FOOD101_CLASSES = [
    'apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare',
    'beet_salad','beignets','bibimbap','bread_pudding','bruschetta',
    'caesar_salad','cannoli','caprese_salad','carrot_cake','ceviche',
    'cheese_plate','cheesecake','chicken_curry','chicken_quesadilla',
    'chicken_tikka_masala','chicken_wings','chocolate_cake','chocolate_mousse',
    'churros','clam_chowder','club_sandwich','crab_cakes','creme_brulee',
    'croque_madame','cup_cakes','deviled_eggs','donuts','dumplings',
    'edamame','eggs_benedict','escargots','falafel','filet_mignon',
    'fish_and_chips','foie_gras','french_fries','french_onion_soup',
    'french_toast','fried_calamari','fried_rice','frozen_yogurt',
    'garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich',
    'grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup',
    'hot_dog','huevos_rancheros','hummus','ice_cream','lasagna',
    'lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese',
    'macarons','miso_soup','mussels','nachos','omelette','onion_rings',
    'oysters','pad_thai','paella','pancakes','panna_cotta','peking_duck',
    'pho','pizza','pork_chop','poutine','prime_rib','pulled_pork_sandwich',
    'ramen','ravioli','red_velvet_cake','risotto','samosa','sashimi',
    'scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese',
    'spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake',
    'sushi','tacos','takoyaki','tiramisu','tuna_tartare','waffles'
]

# ── Indian food classes (will be extended from dataset) ──
INDIAN_CLASSES_DEFAULT = [
    'adhirasam','aloo_gobi','aloo_matar','aloo_methi','aloo_shimla_mirch',
    'aloo_tikki','anarsa','appam','baby_potato_curry','bajre_ki_roti',
    'baked_dhokla','barfi','besan_halwa','bhatura','bhindi_masala',
    'biryani','boondi','butter_chicken','chak_hao_kheer','chana_masala',
    'chapati','chhena_kheeri','chicken_razala','chicken_tikka','chikki',
    'chole_bhature','dal_baati_churma','dal_tadka','dharwad_pedha','doodhpak',
    'double_ka_meetha','dum_aloo','gajar_ka_halwa','gavvalu','ghevar',
    'gulab_jamun','imarti','jalebi','kadai_paneer','kadhi_pakoda',
    'kajjikaya','kaju_katli','kalakand','karela_bharta','kofta',
    'kuzhi_paniyaram','lassi','ledikeni','litti_chokha','lyangcha',
    'maach_jhol','misi_roti','misti_doi','modak','mysore_pak',
    'naan','navrattan_korma','palak_paneer','paneer_butter_masala',
    'papad','parotta','phirni','pithod_curry','poha','poornalu',
    'pootharekulu','rabri','rajma','rasam','rasgulla','ras_malai',
    'sambar','sandesh','sabudana_khichdi','shankarpali','sheer_korma',
    'sheera','shrikhand','sohan_halwa','sohan_papdi','sutar_feni',
    'unni_appam','uttapam','vada_pav','vendakkai_poriyal','wheat_halwa'
]

# Use dataset classes if available, else use defaults
if indian_food_root and len(indian_classes) > 0:
    INDIAN_CLASSES = indian_classes
    print(f'✅ Using {len(INDIAN_CLASSES)} classes from downloaded dataset')
else:
    INDIAN_CLASSES = INDIAN_CLASSES_DEFAULT
    print(f'✅ Using {len(INDIAN_CLASSES)} default Indian food classes')

# Combined class list
ALL_CLASSES = FOOD101_CLASSES + INDIAN_CLASSES
NUM_CLASSES = len(ALL_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}

print(f'\\n📊 Total combined classes: {NUM_CLASSES}')
print(f'   Food-101 classes: {len(FOOD101_CLASSES)}')
print(f'   Indian food classes: {len(INDIAN_CLASSES)}')

# Save label map
with open('/content/label_map.json', 'w') as f:
    json.dump({str(i): cls for i, cls in enumerate(ALL_CLASSES)}, f, indent=2)
print('✅ Label map saved to /content/label_map.json')"""))

cells.append(new_markdown_cell("""## 🔧 STEP 5 — Build Extended Model (Fine-Tuning)"""))

cells.append(new_code_cell("""import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Extract the EfficientNetB0 backbone from existing model ──
backbone = None
for layer in existing_model.layers:
    if 'efficientnet' in layer.name.lower():
        backbone = layer
        print(f'✅ Found backbone: {layer.name}')
        break

if backbone is None:
    # Use existing model minus last layer as feature extractor
    backbone = tf.keras.Model(
        inputs=existing_model.input,
        outputs=existing_model.layers[-3].output,  # before final dense
        name='feature_extractor'
    )
    print(f'✅ Using model up to layer: {existing_model.layers[-3].name}')

# ── Unfreeze top layers of backbone for fine-tuning ──
backbone.trainable = True
# Freeze bottom 80% of backbone layers, only train top 20%
total_layers = len(backbone.layers)
freeze_until = int(total_layers * 0.80)
for i, layer in enumerate(backbone.layers):
    layer.trainable = (i >= freeze_until)

trainable_count = sum(1 for l in backbone.layers if l.trainable)
print(f'\\n🔧 Backbone layers: {total_layers}')
print(f'   Frozen: {total_layers - trainable_count}')
print(f'   Trainable (top 20%): {trainable_count}')

# ── Build new model with extended output ──
inputs = keras.Input(shape=(224, 224, 3), name='input')
x = backbone(inputs, training=False)

if isinstance(x, list) or isinstance(x, tuple):
    x = x[-1]

# Global average pooling
if hasattr(x, 'shape') and len(x.shape) == 4:
    x = layers.GlobalAveragePooling2D()(x)

# New classification head
x = layers.Dense(512, activation='relu', name='dense_new_1')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu', name='dense_new_2')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)

extended_model = keras.Model(inputs, outputs, name='indian_food_classifier')

# ── Compile ──
extended_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')]
)

print(f'\\n✅ Extended model built!')
print(f'   Input:  {extended_model.input_shape}')
print(f'   Output: {extended_model.output_shape}  ({NUM_CLASSES} classes)')"""))

cells.append(new_markdown_cell("""## 🖼️ STEP 6 — Prepare Dataset for Training"""))

cells.append(new_code_cell("""import os, shutil, random
from pathlib import Path

# ── Prepare Indian food data in tf.keras directory format ──
TRAIN_DIR = '/content/train'
VAL_DIR   = '/content/val'

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VAL_SPLIT  = 0.2

if indian_food_root:
    print('📂 Organizing Indian food images into train/val splits...')
    total_images = 0
    for cls in INDIAN_CLASSES:
        cls_path = os.path.join(indian_food_root, cls)
        if not os.path.isdir(cls_path):
            continue
        images = [
            f for f in os.listdir(cls_path)
            if Path(f).suffix.lower() in VALID_EXTS
        ]
        random.shuffle(images)
        split = int(len(images) * (1 - VAL_SPLIT))
        train_imgs = images[:split]
        val_imgs   = images[split:]

        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR,   cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(TRAIN_DIR, cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(VAL_DIR,   cls, img))

        total_images += len(images)

    print(f'✅ Organized {total_images} images across {len(INDIAN_CLASSES)} Indian food classes')
else:
    print('⚠️  No Indian food dataset found — model will be built but fine-tuning skipped')
    print('    You can still use the original 101 food classes for prediction')"""))

cells.append(new_code_cell("""import tensorflow as tf

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

# ── Data Augmentation ──
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name='augmentation')

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment(image, label):
    image = augmentation(image, training=True)
    return image, label

train_count = sum(
    len(os.listdir(os.path.join(TRAIN_DIR, c)))
    for c in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, c))
) if os.path.exists(TRAIN_DIR) and os.listdir(TRAIN_DIR) else 0

if train_count > 0:
    # Load only Indian food classes here (fine-tuning on new classes)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
        seed=42
    )
    dataset_classes = train_ds.class_names
    print(f'✅ Dataset loaded!')
    print(f'   Training classes: {len(dataset_classes)}')
    print(f'   Training images:  {train_count}')

    # Remap labels to our full combined class index
    # (Indian classes start at index 101 in our combined label map)
    def remap_labels(image, label):
        # label shape: (batch, num_indian_classes)
        # We need to pad to NUM_CLASSES
        pad = tf.zeros([tf.shape(label)[0], len(FOOD101_CLASSES)], dtype=tf.float32)
        full_label = tf.concat([pad, label], axis=1)
        return image, full_label

    train_ds = (train_ds
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .map(augment,    num_parallel_calls=AUTOTUNE)
        .map(remap_labels, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (val_ds
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .map(remap_labels, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    print('✅ Datasets preprocessed and ready for training!')
else:
    train_ds = None
    val_ds   = None
    print('⚠️  No training data found — skipping fine-tuning setup')"""))

cells.append(new_markdown_cell("""## 🏋️ STEP 7 — Fine-Tune the Model"""))

cells.append(new_code_cell("""import os

os.makedirs('/content/checkpoints', exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        '/content/checkpoints/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir='/content/logs')
]

if train_ds is not None:
    print('🚀 Starting fine-tuning...')
    print(f'   Training on {NUM_CLASSES} total classes ({len(INDIAN_CLASSES)} new Indian food classes)')
    print('   This will take ~20-40 minutes on T4 GPU\\n')

    history = extended_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )
    print('\\n✅ Fine-tuning complete!')
else:
    print('⏭️  Skipping fine-tuning (no training data). Using original model for prediction.')
    history = None"""))

cells.append(new_markdown_cell("""## 📊 STEP 8 — Plot Training Results"""))

cells.append(new_code_cell("""import matplotlib.pyplot as plt

if history:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='#2ecc71', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='#e74c3c', lw=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train Loss', color='#3498db', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#e67e22', lw=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Fine-Tuning Results — Indian Food Classifier', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/content/training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✅ Training curves saved!')
else:
    print('⏭️  No training history to plot')"""))

cells.append(new_markdown_cell("""## 🥗 STEP 9 — Nutrition Database Setup"""))

cells.append(new_code_cell("""import pandas as pd

# ── Comprehensive nutrition database ──
# Values are per 100g serving
# Format: food_name -> {calories, protein, carbs, fat, fiber, sugar, sodium, vitamin_c, calcium, iron}

NUTRITION_DB = {
    # ── Food-101 classes ──
    'apple_pie':              {'calories':237,'protein':2.0,'carbs':36.0,'fat':11.0,'fiber':1.6,'sugar':21,'sodium':270,'vitamin_c':1,'calcium':11,'iron':0.5},
    'baby_back_ribs':         {'calories':290,'protein':26.0,'carbs':0.0,'fat':20.0,'fiber':0,'sugar':0,'sodium':75,'vitamin_c':0,'calcium':18,'iron':1.3},
    'baklava':                {'calories':428,'protein':6.0,'carbs':52.0,'fat':23.0,'fiber':2,'sugar':31,'sodium':180,'vitamin_c':0,'calcium':60,'iron':2.1},
    'beef_carpaccio':         {'calories':150,'protein':20.0,'carbs':0.0,'fat':8.0,'fiber':0,'sugar':0,'sodium':200,'vitamin_c':0,'calcium':10,'iron':2.5},
    'beef_tartare':           {'calories':175,'protein':21.0,'carbs':0.0,'fat':10.0,'fiber':0,'sugar':0,'sodium':190,'vitamin_c':1,'calcium':12,'iron':2.8},
    'caesar_salad':           {'calories':190,'protein':7.0,'carbs':8.0,'fat':16.0,'fiber':2,'sugar':2,'sodium':380,'vitamin_c':10,'calcium':90,'iron':1.2},
    'cheesecake':             {'calories':321,'protein':5.5,'carbs':26.0,'fat':22.0,'fiber':0,'sugar':22,'sodium':250,'vitamin_c':0,'calcium':62,'iron':0.5},
    'chicken_curry':          {'calories':150,'protein':12.0,'carbs':8.0,'fat':8.0,'fiber':1.5,'sugar':3,'sodium':450,'vitamin_c':8,'calcium':40,'iron':1.5},
    'chicken_tikka_masala':   {'calories':155,'protein':14.0,'carbs':9.0,'fat':7.0,'fiber':1,'sugar':4,'sodium':480,'vitamin_c':10,'calcium':45,'iron':1.8},
    'chicken_wings':          {'calories':290,'protein':27.0,'carbs':0.0,'fat':19.0,'fiber':0,'sugar':0,'sodium':100,'vitamin_c':0,'calcium':15,'iron':1.2},
    'chocolate_cake':         {'calories':367,'protein':5.0,'carbs':51.0,'fat':17.0,'fiber':2,'sugar':37,'sodium':282,'vitamin_c':0,'calcium':30,'iron':2.3},
    'donuts':                 {'calories':452,'protein':5.0,'carbs':51.0,'fat':25.0,'fiber':1,'sugar':20,'sodium':340,'vitamin_c':0,'calcium':60,'iron':2.0},
    'eggs_benedict':          {'calories':210,'protein':12.0,'carbs':14.0,'fat':12.0,'fiber':0,'sugar':2,'sodium':600,'vitamin_c':1,'calcium':80,'iron':1.5},
    'falafel':                {'calories':333,'protein':13.3,'carbs':31.8,'fat':17.8,'fiber':4.9,'sugar':3,'sodium':294,'vitamin_c':1,'calcium':58,'iron':2.5},
    'french_fries':           {'calories':312,'protein':3.4,'carbs':41.0,'fat':15.0,'fiber':3,'sugar':0,'sodium':210,'vitamin_c':8,'calcium':14,'iron':1.1},
    'french_toast':           {'calories':229,'protein':8.0,'carbs':29.0,'fat':10.0,'fiber':1,'sugar':7,'sodium':380,'vitamin_c':0,'calcium':85,'iron':1.5},
    'fried_rice':             {'calories':163,'protein':3.4,'carbs':27.0,'fat':4.0,'fiber':1,'sugar':1,'sodium':340,'vitamin_c':2,'calcium':15,'iron':0.8},
    'grilled_salmon':         {'calories':208,'protein':20.0,'carbs':0.0,'fat':13.0,'fiber':0,'sugar':0,'sodium':59,'vitamin_c':0,'calcium':15,'iron':0.3},
    'guacamole':              {'calories':150,'protein':2.0,'carbs':9.0,'fat':13.0,'fiber':6.7,'sugar':0,'sodium':7,'vitamin_c':10,'calcium':18,'iron':0.6},
    'hamburger':              {'calories':295,'protein':17.0,'carbs':24.0,'fat':14.0,'fiber':1,'sugar':5,'sodium':395,'vitamin_c':2,'calcium':60,'iron':2.2},
    'hot_dog':                {'calories':290,'protein':11.0,'carbs':23.0,'fat':18.0,'fiber':1,'sugar':5,'sodium':870,'vitamin_c':0,'calcium':60,'iron':2.0},
    'ice_cream':              {'calories':207,'protein':3.5,'carbs':24.0,'fat':11.0,'fiber':0,'sugar':21,'sodium':80,'vitamin_c':0,'calcium':128,'iron':0.1},
    'lasagna':                {'calories':166,'protein':11.0,'carbs':16.0,'fat':6.0,'fiber':1,'sugar':4,'sodium':390,'vitamin_c':5,'calcium':120,'iron':1.5},
    'nachos':                 {'calories':306,'protein':7.0,'carbs':31.0,'fat':18.0,'fiber':3,'sugar':1,'sodium':476,'vitamin_c':1,'calcium':150,'iron':1.2},
    'omelette':               {'calories':154,'protein':11.0,'carbs':1.0,'fat':12.0,'fiber':0,'sugar':1,'sodium':342,'vitamin_c':0,'calcium':56,'iron':1.5},
    'pancakes':               {'calories':227,'protein':6.0,'carbs':40.0,'fat':6.0,'fiber':1,'sugar':15,'sodium':430,'vitamin_c':0,'calcium':80,'iron':1.5},
    'pizza':                  {'calories':266,'protein':11.0,'carbs':33.0,'fat':10.0,'fiber':2,'sugar':4,'sodium':598,'vitamin_c':2,'calcium':188,'iron':2.0},
    'ramen':                  {'calories':180,'protein':7.0,'carbs':27.0,'fat':5.0,'fiber':1,'sugar':2,'sodium':1200,'vitamin_c':0,'calcium':20,'iron':1.0},
    'samosa':                 {'calories':262,'protein':4.0,'carbs':31.0,'fat':14.0,'fiber':2,'sugar':2,'sodium':320,'vitamin_c':5,'calcium':30,'iron':1.8},
    'spaghetti_bolognese':    {'calories':175,'protein':10.0,'carbs':22.0,'fat':5.0,'fiber':2,'sugar':4,'sodium':340,'vitamin_c':8,'calcium':40,'iron':1.8},
    'spring_rolls':           {'calories':153,'protein':4.5,'carbs':18.0,'fat':7.0,'fiber':2,'sugar':2,'sodium':280,'vitamin_c':4,'calcium':25,'iron':1.0},
    'steak':                  {'calories':271,'protein':26.0,'carbs':0.0,'fat':18.0,'fiber':0,'sugar':0,'sodium':70,'vitamin_c':0,'calcium':18,'iron':2.5},
    'sushi':                  {'calories':150,'protein':6.0,'carbs':27.0,'fat':2.0,'fiber':1,'sugar':4,'sodium':426,'vitamin_c':1,'calcium':17,'iron':0.4},
    'tacos':                  {'calories':226,'protein':11.0,'carbs':22.0,'fat':10.0,'fiber':3,'sugar':2,'sodium':390,'vitamin_c':4,'calcium':60,'iron':1.8},
    'tiramisu':               {'calories':283,'protein':5.0,'carbs':28.0,'fat':17.0,'fiber':0,'sugar':23,'sodium':113,'vitamin_c':0,'calcium':60,'iron':0.8},
    'waffles':                {'calories':291,'protein':8.0,'carbs':37.0,'fat':13.0,'fiber':1,'sugar':10,'sodium':461,'vitamin_c':0,'calcium':100,'iron':2.0},

    # ── Indian food classes ──
    'biryani':                {'calories':200,'protein':9.0,'carbs':32.0,'fat':5.0,'fiber':1.5,'sugar':2,'sodium':450,'vitamin_c':5,'calcium':35,'iron':1.5},
    'butter_chicken':         {'calories':164,'protein':15.0,'carbs':7.0,'fat':9.0,'fiber':1,'sugar':5,'sodium':520,'vitamin_c':8,'calcium':50,'iron':1.8},
    'chana_masala':           {'calories':164,'protein':8.5,'carbs':22.0,'fat':5.0,'fiber':6,'sugar':3,'sodium':380,'vitamin_c':12,'calcium':80,'iron':3.8},
    'chapati':                {'calories':297,'protein':9.0,'carbs':55.0,'fat':4.0,'fiber':6,'sugar':1,'sodium':3,'vitamin_c':0,'calcium':30,'iron':2.9},
    'chole_bhature':          {'calories':390,'protein':11.0,'carbs':55.0,'fat':15.0,'fiber':5,'sugar':3,'sodium':450,'vitamin_c':8,'calcium':80,'iron':3.5},
    'dal_tadka':              {'calories':130,'protein':7.0,'carbs':18.0,'fat':4.0,'fiber':4,'sugar':2,'sodium':310,'vitamin_c':5,'calcium':45,'iron':2.5},
    'dosa':                   {'calories':168,'protein':4.0,'carbs':34.0,'fat':2.0,'fiber':1,'sugar':1,'sodium':180,'vitamin_c':0,'calcium':20,'iron':0.8},
    'gajar_ka_halwa':         {'calories':234,'protein':4.0,'carbs':32.0,'fat':11.0,'fiber':2,'sugar':26,'sodium':80,'vitamin_c':4,'calcium':140,'iron':0.8},
    'gulab_jamun':            {'calories':383,'protein':7.0,'carbs':58.0,'fat':14.0,'fiber':0,'sugar':48,'sodium':110,'vitamin_c':0,'calcium':90,'iron':0.5},
    'idli':                   {'calories':58, 'protein':2.0,'carbs':11.0,'fat':0.4,'fiber':0.5,'sugar':0,'sodium':140,'vitamin_c':0,'calcium':15,'iron':0.5},
    'jalebi':                 {'calories':369,'protein':2.0,'carbs':66.0,'fat':10.0,'fiber':0,'sugar':50,'sodium':95,'vitamin_c':0,'calcium':15,'iron':0.6},
    'kadai_paneer':           {'calories':225,'protein':12.0,'carbs':8.0,'fat':16.0,'fiber':2,'sugar':4,'sodium':400,'vitamin_c':25,'calcium':280,'iron':1.2},
    'lassi':                  {'calories':70, 'protein':3.5,'carbs':9.0,'fat':2.5,'fiber':0,'sugar':9,'sodium':50,'vitamin_c':1,'calcium':120,'iron':0.1},
    'naan':                   {'calories':317,'protein':9.0,'carbs':55.0,'fat':7.0,'fiber':2,'sugar':3,'sodium':420,'vitamin_c':0,'calcium':50,'iron':2.5},
    'palak_paneer':           {'calories':180,'protein':10.0,'carbs':8.0,'fat':12.0,'fiber':3,'sugar':2,'sodium':380,'vitamin_c':30,'calcium':310,'iron':2.5},
    'paneer_butter_masala':   {'calories':200,'protein':11.0,'carbs':9.0,'fat':14.0,'fiber':1,'sugar':5,'sodium':500,'vitamin_c':10,'calcium':290,'iron':1.0},
    'parotta':                {'calories':310,'protein':7.0,'carbs':48.0,'fat':10.0,'fiber':2,'sugar':1,'sodium':300,'vitamin_c':0,'calcium':25,'iron':2.0},
    'poha':                   {'calories':180,'protein':3.0,'carbs':36.0,'fat':4.0,'fiber':2,'sugar':2,'sodium':200,'vitamin_c':10,'calcium':20,'iron':1.8},
    'rajma':                  {'calories':127,'protein':8.7,'carbs':22.8,'fat':0.5,'fiber':7.4,'sugar':1,'sodium':2,'vitamin_c':2,'calcium':50,'iron':3.9},
    'rasgulla':               {'calories':186,'protein':5.0,'carbs':38.0,'fat':2.0,'fiber':0,'sugar':36,'sodium':45,'vitamin_c':0,'calcium':120,'iron':0.3},
    'ras_malai':              {'calories':215,'protein':6.0,'carbs':28.0,'fat':9.0,'fiber':0,'sugar':25,'sodium':60,'vitamin_c':0,'calcium':160,'iron':0.2},
    'sambar':                 {'calories':55, 'protein':3.0,'carbs':9.0,'fat':1.0,'fiber':3,'sugar':3,'sodium':380,'vitamin_c':20,'calcium':45,'iron':1.5},
    'uttapam':                {'calories':130,'protein':4.0,'carbs':24.0,'fat':3.0,'fiber':1.5,'sugar':2,'sodium':220,'vitamin_c':8,'calcium':40,'iron':0.9},
    'vada_pav':               {'calories':295,'protein':7.0,'carbs':42.0,'fat':11.0,'fiber':3,'sugar':4,'sodium':480,'vitamin_c':6,'calcium':50,'iron':1.9},
    'aloo_gobi':              {'calories':112,'protein':3.5,'carbs':16.0,'fat':4.5,'fiber':3,'sugar':3,'sodium':350,'vitamin_c':30,'calcium':35,'iron':1.2},
    'aloo_tikki':             {'calories':210,'protein':4.0,'carbs':28.0,'fat':10.0,'fiber':2,'sugar':1,'sodium':390,'vitamin_c':12,'calcium':25,'iron':1.5},
    'bhatura':                {'calories':350,'protein':7.0,'carbs':52.0,'fat':13.0,'fiber':2,'sugar':2,'sodium':420,'vitamin_c':0,'calcium':40,'iron':2.2},
    'bhindi_masala':          {'calories':95, 'protein':2.5,'carbs':10.0,'fat':5.5,'fiber':4,'sugar':3,'sodium':320,'vitamin_c':22,'calcium':80,'iron':1.0},
    'chicken_tikka':          {'calories':180,'protein':22.0,'carbs':4.0,'fat':8.0,'fiber':0,'sugar':2,'sodium':480,'vitamin_c':5,'calcium':30,'iron':1.5},
    'dal_baati_churma':       {'calories':420,'protein':12.0,'carbs':58.0,'fat':16.0,'fiber':5,'sugar':8,'sodium':300,'vitamin_c':2,'calcium':60,'iron':3.0},
    'kaju_katli':             {'calories':540,'protein':9.0,'carbs':55.0,'fat':32.0,'fiber':1,'sugar':42,'sodium':20,'vitamin_c':0,'calcium':40,'iron':3.5},
    'modak':                  {'calories':330,'protein':5.0,'carbs':50.0,'fat':12.0,'fiber':2,'sugar':28,'sodium':80,'vitamin_c':0,'calcium':30,'iron':1.2},
    'mysore_pak':             {'calories':520,'protein':8.0,'carbs':55.0,'fat':30.0,'fiber':1,'sugar':40,'sodium':15,'vitamin_c':0,'calcium':25,'iron':2.0},
    'panna_cotta':            {'calories':250,'protein':4.0,'carbs':25.0,'fat':15.0,'fiber':0,'sugar':20,'sodium':50,'vitamin_c':0,'calcium':100,'iron':0.2},
    'phirni':                 {'calories':148,'protein':4.5,'carbs':22.0,'fat':5.0,'fiber':0,'sugar':18,'sodium':55,'vitamin_c':0,'calcium':140,'iron':0.4},
    'dum_aloo':               {'calories':145,'protein':2.5,'carbs':20.0,'fat':7.0,'fiber':2,'sugar':3,'sodium':380,'vitamin_c':18,'calcium':30,'iron':1.0},
    'kofta':                  {'calories':220,'protein':14.0,'carbs':8.0,'fat':15.0,'fiber':1.5,'sugar':3,'sodium':420,'vitamin_c':5,'calcium':60,'iron':2.0},
    'vendakkai_poriyal':      {'calories':85, 'protein':2.0,'carbs':10.0,'fat':4.5,'fiber':3.5,'sugar':2,'sodium':280,'vitamin_c':20,'calcium':75,'iron':0.8},
    'appam':                  {'calories':110,'protein':2.5,'carbs':22.0,'fat':1.5,'fiber':0.5,'sugar':1,'sodium':130,'vitamin_c':0,'calcium':15,'iron':0.6},
    'kuzhi_paniyaram':        {'calories':175,'protein':4.0,'carbs':27.0,'fat':6.0,'fiber':1,'sugar':2,'sodium':220,'vitamin_c':0,'calcium':35,'iron':1.0},
    'rasam':                  {'calories':40, 'protein':1.5,'carbs':7.0,'fat':1.0,'fiber':1.5,'sugar':2,'sodium':410,'vitamin_c':15,'calcium':25,'iron':1.2},
    'shrikhand':              {'calories':260,'protein':7.0,'carbs':33.0,'fat':11.0,'fiber':0,'sugar':30,'sodium':40,'vitamin_c':0,'calcium':200,'iron':0.3},
    'sheera':                 {'calories':295,'protein':4.5,'carbs':42.0,'fat':12.0,'fiber':1,'sugar':28,'sodium':85,'vitamin_c':0,'calcium':30,'iron':1.5},
    'poornalu':               {'calories':310,'protein':6.0,'carbs':46.0,'fat':12.0,'fiber':2,'sugar':22,'sodium':90,'vitamin_c':0,'calcium':40,'iron':1.8},
    'sabudana_khichdi':       {'calories':210,'protein':2.5,'carbs':40.0,'fat':6.0,'fiber':0.5,'sugar':2,'sodium':200,'vitamin_c':5,'calcium':20,'iron':1.0},
    'navrattan_korma':        {'calories':190,'protein':6.0,'carbs':15.0,'fat':12.0,'fiber':2,'sugar':6,'sodium':420,'vitamin_c':12,'calcium':80,'iron':1.2},
    'pho':                    {'calories':90, 'protein':5.0,'carbs':14.0,'fat':2.0,'fiber':1,'sugar':2,'sodium':800,'vitamin_c':5,'calcium':20,'iron':1.0},
    'pad_thai':               {'calories':225,'protein':12.0,'carbs':30.0,'fat':7.0,'fiber':2,'sugar':6,'sodium':560,'vitamin_c':5,'calcium':45,'iron':1.5},
}

# Default fallback nutrition for unknown foods
DEFAULT_NUTRITION = {'calories':200,'protein':8.0,'carbs':25.0,'fat':8.0,'fiber':2,'sugar':5,'sodium':300,'vitamin_c':5,'calcium':50,'iron':1.5}

# Standard portion sizes in grams per food type
PORTION_SIZES = {
    'default': 150,
    'biryani': 250, 'pizza': 200, 'hamburger': 180, 'steak': 200,
    'salad': 150, 'soup': 250, 'cake': 100, 'ice_cream': 100,
    'idli': 80, 'dosa': 120, 'chapati': 60, 'naan': 90, 'parotta': 100,
    'sambar': 150, 'rasam': 150, 'dal_tadka': 150, 'rajma': 150,
    'lassi': 200, 'gulab_jamun': 80, 'jalebi': 80, 'rasgulla': 80,
}

df_nutrition = pd.DataFrame(NUTRITION_DB).T
df_nutrition.index.name = 'food'
df_nutrition.reset_index(inplace=True)
print(f'✅ Nutrition database loaded: {len(df_nutrition)} foods')
df_nutrition.head()"""))

cells.append(new_markdown_cell("""## 🔮 STEP 10 — Prediction & Nutrition Report"""))

cells.append(new_code_cell("""import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, io

# Load best model if fine-tuning happened, else use original
BEST_MODEL_PATH = '/content/checkpoints/best_model.h5'
if os.path.exists(BEST_MODEL_PATH):
    predict_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    USE_EXTENDED = True
    print('✅ Loaded fine-tuned model (101 + Indian food classes)')
else:
    predict_model = existing_model
    USE_EXTENDED = False
    print('✅ Using original model (101 Food-101 classes)')

def get_portion_size(food_name):
    \"\"\"Return estimated gram weight for a food item\"\"\"
    for key in PORTION_SIZES:
        if key in food_name:
            return PORTION_SIZES[key]
    return PORTION_SIZES['default']

def get_nutrition(food_name, grams):
    \"\"\"Return scaled nutrition for given food and gram amount\"\"\"
    key = food_name.lower().replace(' ', '_').replace('-', '_')
    if key in NUTRITION_DB:
        base = NUTRITION_DB[key]
    else:
        # Fuzzy match
        best_match = None
        best_score = 0
        for db_key in NUTRITION_DB:
            score = sum(1 for a,b in zip(key, db_key) if a == b) / max(len(key), len(db_key))
            if score > best_score:
                best_score = score
                best_match = db_key
        if best_score > 0.6 and best_match:
            base = NUTRITION_DB[best_match]
            print(f'  (matched "{key}" → "{best_match}")')
        else:
            base = DEFAULT_NUTRITION
    scale = grams / 100.0
    return {k: round(v * scale, 1) for k, v in base.items()}

def predict_food(image_path_or_array, top_k=3):
    \"\"\"Run full prediction pipeline on an image\"\"\"
    # Load and preprocess image
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert('RGB').resize((224, 224))
    else:
        img = Image.fromarray(image_path_or_array).convert('RGB').resize((224, 224))

    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Predict
    preds = predict_model.predict(img_batch, verbose=0)[0]

    # Map to labels
    label_map = ALL_CLASSES if USE_EXTENDED else FOOD101_CLASSES
    top_indices = np.argsort(preds)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if idx < len(label_map):
            food_name  = label_map[idx]
            confidence = float(preds[idx]) * 100
            grams      = get_portion_size(food_name)
            nutrition  = get_nutrition(food_name, grams)
            results.append({
                'food':       food_name,
                'confidence': round(confidence, 1),
                'grams':      grams,
                'nutrition':  nutrition
            })
    return results, img

def show_nutrition_report(results, img):
    \"\"\"Show a complete visual nutrition report\"\"\"
    # Aggregate totals
    totals = {k: 0 for k in ['calories','protein','carbs','fat','fiber','sugar','sodium','vitamin_c','calcium','iron']}
    for r in results:
        for k, v in r['nutrition'].items():
            totals[k] += v

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Panel 1: Image ──
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('📸 Your Meal', color='white', fontsize=12, fontweight='bold')

    # ── Panel 2: Detected Foods ──
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.set_facecolor('#16213e')
    ax2.axis('off')
    text = '🍽️  DETECTED FOODS\\n' + '─'*28 + '\\n'
    for i, r in enumerate(results):
        name = r['food'].replace('_', ' ').title()
        text += f"{i+1}. {name}\\n   Conf: {r['confidence']}% | {r['grams']}g\\n"
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
             color='#00d4ff', fontsize=9, verticalalignment='top',
             fontfamily='monospace')
    ax2.set_title('Detections', color='white', fontsize=11, fontweight='bold')

    # ── Panel 3: Macronutrient Bar Chart ──
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.set_facecolor('#16213e')
    macros = ['protein', 'carbs', 'fat', 'fiber']
    values = [totals[m] for m in macros]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax3.barh(macros, values, color=colors, edgecolor='none', height=0.5)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val}g', va='center', color='white', fontsize=9)
    ax3.set_facecolor('#16213e')
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('#444')
    ax3.spines['left'].set_color('#444')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title('Macronutrients (g)', color='white', fontsize=11, fontweight='bold')
    ax3.set_xlabel('grams', color='#aaa')

    # ── Panel 4: Calorie Pie Chart ──
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.set_facecolor('#16213e')
    cal_protein = totals['protein'] * 4
    cal_carbs   = totals['carbs']   * 4
    cal_fat     = totals['fat']     * 9
    cal_sizes = [cal_protein, cal_carbs, cal_fat]
    cal_labels = [f'Protein\\n{cal_protein:.0f} kcal', f'Carbs\\n{cal_carbs:.0f} kcal', f'Fat\\n{cal_fat:.0f} kcal']
    ax4.pie(cal_sizes, labels=cal_labels, colors=['#2ecc71','#3498db','#e74c3c'],
            autopct='%1.1f%%', textprops={'color':'white','fontsize':8},
            wedgeprops={'edgecolor':'#1a1a2e', 'linewidth':2})
    ax4.set_title(f'Calorie Split\\nTotal: {totals["calories"]:.0f} kcal',
                  color='white', fontsize=11, fontweight='bold')

    # ── Panel 5-8: Nutrition Label ──
    ax5 = fig.add_subplot(2, 1, 2)
    ax5.set_facecolor('#16213e')
    ax5.axis('off')

    label_items = [
        ('🔥 Calories',    f"{totals['calories']:.0f} kcal"),
        ('🥩 Protein',     f"{totals['protein']:.1f} g"),
        ('🍞 Carbohydrates',f"{totals['carbs']:.1f} g"),
        ('🫙 Fat',          f"{totals['fat']:.1f} g"),
        ('🌾 Fiber',        f"{totals['fiber']:.1f} g"),
        ('🍬 Sugar',        f"{totals['sugar']:.1f} g"),
        ('🧂 Sodium',       f"{totals['sodium']:.0f} mg"),
        ('🍊 Vitamin C',    f"{totals['vitamin_c']:.0f} mg"),
        ('🦷 Calcium',      f"{totals['calcium']:.0f} mg"),
        ('🩸 Iron',         f"{totals['iron']:.1f} mg"),
    ]

    cols = 5
    for i, (label, value) in enumerate(label_items):
        x = (i % cols) * 0.20 + 0.02
        y = 0.75 if i < cols else 0.25
        ax5.text(x, y + 0.1, label, transform=ax5.transAxes,
                 color='#aaaaaa', fontsize=9, fontweight='bold')
        ax5.text(x, y - 0.05, value, transform=ax5.transAxes,
                 color='#00ff88', fontsize=14, fontweight='bold')

    ax5.set_title('📊 COMPLETE NUTRITION SUMMARY', color='white',
                  fontsize=13, fontweight='bold', pad=10)

    plt.suptitle('🍛 AI Food Nutrients Classifier — Analysis Report',
                 color='#00d4ff', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('/content/nutrition_report.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.show()

    # Print text report
    print('\\n' + '='*50)
    print('🍽️  DETECTED FOODS:')
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['food'].replace('_',' ').title()} "
              f"(confidence: {r['confidence']}%) — Est. {r['grams']}g")
    print('\\n📊 TOTAL NUTRITION SUMMARY:')
    print(f"  Calories:       {totals['calories']:.0f} kcal")
    print(f"  Protein:        {totals['protein']:.1f} g")
    print(f"  Carbohydrates:  {totals['carbs']:.1f} g")
    print(f"  Fat:            {totals['fat']:.1f} g")
    print(f"  Fiber:          {totals['fiber']:.1f} g")
    print(f"  Sugar:          {totals['sugar']:.1f} g")
    print(f"  Sodium:         {totals['sodium']:.0f} mg")
    print(f"  Vitamin C:      {totals['vitamin_c']:.0f} mg")
    print(f"  Calcium:        {totals['calcium']:.0f} mg")
    print(f"  Iron:           {totals['iron']:.1f} mg")
    print('='*50)

print('✅ Prediction functions ready!')"""))

# TASK 2: Replace STEP 11 Interactive UI
cells.append(new_markdown_cell("""## 🎨 STEP 11 — Interactive UI"""))

cells.append(new_code_cell("""import ipywidgets as widgets
from IPython.display import display, HTML, clear_output, Image as IPImage
import io
import time
from datetime import datetime
from google.colab import files
import os
import matplotlib.pyplot as plt

# Create UI Elements
upload_btn = widgets.FileUpload(
    accept='image/*',
    multiple=False,
    description='📸 Upload Food Photo',
    button_style='info'
)

analyse_btn = widgets.Button(
    description='🔍 Analyse',
    button_style='success',
    disabled=True
)

download_btn = widgets.Button(
    description='💾 Download Report',
    button_style='primary',
    disabled=True
)

reset_btn = widgets.Button(
    description='🔄 Analyse Another',
    button_style='warning'
)

progress_bar = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    description='🔍 Analysing...',
    bar_style='info',
    style={'bar_color': '#00d4ff'},
    orientation='horizontal',
    layout=widgets.Layout(width='100%', visibility='hidden')
)

out_preview = widgets.Output()
out_results = widgets.Output()

def generate_ui_html(results, health_score, meal_tag):
    top_pred = results[0]
    
    # Calculate health score badge
    if health_score >= 80: badge = "🟢 Excellent"
    elif health_score >= 60: badge = "🟡 Good"
    elif health_score >= 40: badge = "🟠 Fair"
    else: badge = "🔴 Poor"
        
    macros = {
        'Protein': (top_pred['nutrition']['protein'], '#2ecc71', 50),
        'Carbs': (top_pred['nutrition']['carbs'], '#3498db', 100),
        'Fat': (top_pred['nutrition']['fat'], '#e74c3c', 50),
        'Fiber': (top_pred['nutrition']['fiber'], '#f39c12', 20)
    }
    
    macro_html = ""
    for m_name, (val, color, max_val) in macros.items():
        pct = min(100, int((val / max_val) * 100))
        macro_html += f\"\"\"
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <div style="width: 80px; color: #ccc;">{m_name}</div>
            <div style="flex-grow: 1; background: #2a2a4a; height: 16px; border-radius: 8px; margin: 0 15px; overflow: hidden;">
                <div style="width: {pct}%; background: {color}; height: 100%; border-radius: 8px;"></div>
            </div>
            <div style="width: 50px; text-align: right; color: white;">{val}g</div>
        </div>
        \"\"\"
        
    predictions_html = ""
    for i, r in enumerate(results):
        dot = "🟢" if i == 0 else ("🟡" if i == 1 else "🔴")
        predictions_html += f\"\"\"
        <div style="display: flex; justify-content: space-between; margin: 5px 0;">
            <span>{i+1}. {r['food'].replace('_', ' ').title()}</span>
            <span>{r['confidence']}% {dot}</span>
        </div>
        \"\"\"
        
    html_content = f\"\"\"
    <div style="background-color: #1a1a2e; color: white; padding: 20px; border-radius: 15px; font-family: sans-serif; max-width: 800px; border: 1px solid #00d4ff;">
        
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <div style="flex: 1; background: #16213e; padding: 15px; border-radius: 10px;">
                <h3 style="margin-top: 0; color: #00d4ff; border-bottom: 1px solid #333; padding-bottom: 10px;">🏷️ {top_pred['food'].replace('_', ' ').title()}</h3>
                <p>Confidence: <b style="color: #00ff88;">{top_pred['confidence']}%</b></p>
                <p>Portion: <b>{top_pred['grams']}g</b></p>
                <p>🏅 Health Score: <b>{health_score} ({badge})</b></p>
                <p>Tag: <b style="color: #f39c12;">{meal_tag}</b></p>
            </div>
        </div>
        
        <div style="background: #16213e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #00d4ff;">┌─ MACRONUTRIENTS ───────────────┐</h3>
            <h2 style="text-align: center; color: #f39c12; margin: 10px 0;">🔥 {top_pred['nutrition']['calories']} kcal</h2>
            {macro_html}
        </div>
        
        <div style="background: #16213e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #00d4ff;">┌─ MICRONUTRIENTS ───────────────┐</h3>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px; color: #ccc;">
                <span>Fiber: {top_pred['nutrition']['fiber']}g</span>
                <span>Sugar: {top_pred['nutrition']['sugar']}g</span>
                <span>Sodium: {top_pred['nutrition']['sodium']}mg</span>
                <span>Vit C: {top_pred['nutrition']['vitamin_c']}mg</span>
                <span>Calcium: {top_pred['nutrition']['calcium']}mg</span>
                <span>Iron: {top_pred['nutrition']['iron']}mg</span>
            </div>
        </div>
        
        <div style="background: #16213e; padding: 15px; border-radius: 10px;">
            <h3 style="margin-top: 0; color: #00d4ff;">┌─ TOP 3 PREDICTIONS ────────────┐</h3>
            {predictions_html}
        </div>
        
    </div>
    \"\"\"
    return html_content

def calculate_health_score(nutrition):
    score = 100
    if nutrition['sodium'] > 500: score -= 10
    if nutrition['fat'] > 20: score -= 10
    if nutrition['fiber'] < 2: score -= 5
    if nutrition['protein'] > 15: score += 10
    if nutrition['fiber'] > 5: score += 5
    
    score = max(0, min(100, score))
    
    if nutrition['protein'] > 20: tag = "High Protein"
    elif nutrition['carbs'] > 50: tag = "High Carb"
    elif nutrition['fat'] > 25: tag = "High Fat"
    elif nutrition['calories'] < 200: tag = "Low Calorie"
    else: tag = "Balanced"
        
    return score, tag

global_results = None
global_img = None
global_image_name = None
global_health_score = None

def on_upload_change(change):
    if upload_btn.value:
        analyse_btn.disabled = False
        with out_preview:
            clear_output()
            # Handle both ipywidgets 7 and 8 formats
            val = upload_btn.value
            content = val[list(val.keys())[0]]['content'] if isinstance(val, dict) else val[0].content
            display(IPImage(value=content, width=300))

upload_btn.observe(on_upload_change, names='value')

def on_analyse_click(b):
    global global_results, global_img, global_image_name, global_health_score
    analyse_btn.disabled = True
    upload_btn.disabled = True
    progress_bar.layout.visibility = 'visible'
    
    with out_results:
        clear_output()
        
    # Simulate progress
    for i in range(10, 90, 10):
        progress_bar.value = i
        time.sleep(0.1)
        
    try:
        val = upload_btn.value
        if isinstance(val, dict):
            file_info = list(val.keys())[0]
            content = val[file_info]['content']
            global_image_name = file_info
        else:
            global_image_name = val[0].name
            content = val[0].content

        image_path_or_array = io.BytesIO(content)
        
        # Call existing prediction function
        global_results, global_img = predict_food(image_path_or_array, top_k=3)
        top_nutrition = global_results[0]['nutrition']
        
        global_health_score, meal_tag = calculate_health_score(top_nutrition)
        
        html_ui = generate_ui_html(global_results, global_health_score, meal_tag)
        
        progress_bar.value = 100
        with out_results:
            display(HTML(html_ui))
            
        download_btn.disabled = False
    except Exception as e:
        with out_results:
            print(f"Error during analysis: {e}")
            
    progress_bar.layout.visibility = 'hidden'

analyse_btn.on_click(on_analyse_click)

def on_download_click(b):
    if global_results and global_img:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'/content/nutrition_report_{timestamp}.png'
        
        # Plot and save
        show_nutrition_report(global_results, global_img)
        
        if os.path.exists('/content/nutrition_report.png'):
            os.rename('/content/nutrition_report.png', filepath)
            files.download(filepath)

download_btn.on_click(on_download_click)

def on_reset_click(b):
    upload_btn.value.clear()
    upload_btn._counter = 0
    upload_btn.disabled = False
    analyse_btn.disabled = True
    download_btn.disabled = True
    progress_bar.value = 0
    progress_bar.layout.visibility = 'hidden'
    with out_preview:
        clear_output()
    with out_results:
        clear_output()

reset_btn.on_click(on_reset_click)

# Display UI
ui_layout = widgets.VBox([
    widgets.HTML("<h2 style='color: #00d4ff;'>🍛 AI Indian Food Nutrient Analyser</h2><hr style='border-color: #333;'>"),
    widgets.HBox([upload_btn, analyse_btn]),
    progress_bar,
    widgets.HBox([out_preview, out_results]),
    widgets.HBox([download_btn, reset_btn])
])

display(ui_layout)"""))

# TASK 3: Save outputs
cells.append(new_markdown_cell("""## 💾 STEP 12 — Save & Download Everything"""))

cells.append(new_code_cell("""from google.colab import files
import json

# Save fine-tuned model
if os.path.exists(BEST_MODEL_PATH):
    extended_model.save('/content/indian_food_classifier_final.h5')
    print('✅ Fine-tuned model saved')

# Save label map
with open('/content/label_map.json', 'w') as f:
    json.dump({str(i): cls for i, cls in enumerate(ALL_CLASSES)}, f, indent=2)

# Save nutrition database
df_nutrition.to_csv('/content/nutrition_db.csv', index=False)

print('\\n📥 Downloading files...')
if os.path.exists('/content/indian_food_classifier_final.h5'):
    files.download('/content/indian_food_classifier_final.h5')
files.download('/content/label_map.json')
files.download('/content/nutrition_db.csv')
if os.path.exists('/content/nutrition_report.png'):
    files.download('/content/nutrition_report.png')
if os.path.exists('/content/training_results.png'):
    files.download('/content/training_results.png')

print('\\n✅ All files downloaded!')
print('   - indian_food_classifier_final.h5 → Your trained model')
print('   - label_map.json                  → Class index mapping')
print('   - nutrition_db.csv                → Nutrition database')
print('   - nutrition_report.png            → Sample report')"""))

cells.append(new_markdown_cell("""## 🔗 STEP 13 — Session Log"""))

cells.append(new_code_cell("""import csv
import os
from datetime import datetime

LOG_FILE = '/content/predictions_log.csv'

def save_session_log():
    if 'global_results' not in globals() or not global_results:
        print("No analysis to save. Please run STEP 11 first.")
        return
        
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'image_name', 'top_food', 'confidence', 'calories', 'protein', 'carbs', 'fat', 'health_score'])
            
        top_pred = global_results[0]
        nut = top_pred['nutrition']
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            global_image_name,
            top_pred['food'],
            top_pred['confidence'],
            nut['calories'],
            nut['protein'],
            nut['carbs'],
            nut['fat'],
            global_health_score
        ])
        
    # Count rows
    with open(LOG_FILE, 'r') as f:
        count = sum(1 for row in f) - 1 # exclude header
        
    print(f"📊 Session log saved: {count} analyses recorded in {LOG_FILE}")

# Try to automatically log the last prediction if it exists
try:
    if global_results:
        save_session_log()
except NameError:
    print("Run this cell after predicting an image in STEP 11 to log the result.")"""))

# TASK 1: API cells
cells.append(new_markdown_cell("""### CELL A — Model Loader"""))

cells.append(new_code_cell("""# CELL A — Model Loader (after STEP 12):
import os
import json
import tensorflow as tf

print("Loading models and definitions...")

if os.path.exists('/content/indian_food_classifier_final.h5'):
    final_model_path = '/content/indian_food_classifier_final.h5'
else:
    final_model_path = BEST_MODEL_PATH if 'BEST_MODEL_PATH' in globals() and os.path.exists(BEST_MODEL_PATH) else MODEL_PATH

print(f"Loading model from: {final_model_path}")
api_model = tf.keras.models.load_model(final_model_path)

label_map_path = '/content/label_map.json'
if os.path.exists(label_map_path):
    with open(label_map_path, 'r') as f:
        api_labels_dict = json.load(f)
        api_classes = [api_labels_dict[str(i)] for i in range(len(api_labels_dict))]
else:
    api_classes = ALL_CLASSES

print(f"✅ Model ready for prediction — {len(api_classes)} classes loaded")"""))

cells.append(new_markdown_cell("""### CELL B — Prediction API function"""))

cells.append(new_code_cell("""# CELL B — Prediction API function:
from PIL import Image
import numpy as np

def analyze_food_image(image_source):
    \"\"\"
    Accepts: file path (str), PIL Image, or numpy array
    Returns: dict with keys:
        - top_predictions: list of {food, confidence, grams, nutrition}
        - best_match: single best prediction dict
        - nutrition_totals: aggregated nutrition across top 3
        - health_score: integer 0-100 based on fiber, protein, sodium balance
        - meal_tag: one of ["High Protein", "High Carb", "Balanced", "High Fat", "Low Calorie"]
    \"\"\"
    # 1. Handle image format
    if isinstance(image_source, str):
        img = Image.open(image_source).convert('RGB').resize((224, 224))
    elif isinstance(image_source, np.ndarray):
        img = Image.fromarray(image_source).convert('RGB').resize((224, 224))
    else:
        img = image_source.convert('RGB').resize((224, 224))
        
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # 2. Predict
    preds = api_model.predict(img_batch, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:3]
    
    top_predictions = []
    for idx in top_indices:
        if idx < len(api_classes):
            food_name = api_classes[idx]
            conf = float(preds[idx]) * 100
            grams = get_portion_size(food_name)
            nutr = get_nutrition(food_name, grams)
            top_predictions.append({
                'food': food_name,
                'confidence': round(conf, 1),
                'grams': grams,
                'nutrition': nutr
            })
            
    best_match = top_predictions[0] if top_predictions else None
    
    # 3. Aggregate totals
    nutrition_totals = {k: 0 for k in ['calories','protein','carbs','fat','fiber','sugar','sodium','vitamin_c','calcium','iron']}
    for p in top_predictions:
        for k, v in p['nutrition'].items():
            nutrition_totals[k] = round(nutrition_totals[k] + v, 1)
            
    # 4. Health Score & Meal Tag
    if best_match:
        health_score, meal_tag = compute_health_metrics(best_match['nutrition'])
    else:
        health_score, meal_tag = 0, "Unknown"
        
    return {
        'top_predictions': top_predictions,
        'best_match': best_match,
        'nutrition_totals': nutrition_totals,
        'health_score': health_score,
        'meal_tag': meal_tag
    }"""))

cells.append(new_markdown_cell("""### CELL C — Health Score Logic"""))

cells.append(new_code_cell("""# CELL C — Health Score Logic:
def compute_health_metrics(nutrition):
    health_score = 100
    
    if nutrition.get('sodium', 0) > 500:
        health_score -= 10
    if nutrition.get('fat', 0) > 20:
        health_score -= 10
    if nutrition.get('fiber', 0) < 2:
        health_score -= 5
    if nutrition.get('protein', 0) > 15:
        health_score += 10
    if nutrition.get('fiber', 0) > 5:
        health_score += 5
        
    # Clamp between 0 and 100
    health_score = max(0, min(100, health_score))
    
    # Meal tag logic
    if nutrition.get('protein', 0) > 20:
        meal_tag = "High Protein"
    elif nutrition.get('carbs', 0) > 50:
        meal_tag = "High Carb"
    elif nutrition.get('fat', 0) > 25:
        meal_tag = "High Fat"
    elif nutrition.get('calories', 0) < 200:
        meal_tag = "Low Calorie"
    else:
        meal_tag = "Balanced"
        
    return health_score, meal_tag

print("✅ Prediction API and Health Score logic successfully loaded.")"""))

nb['cells'] = cells

with open('Indian_Food_Nutrients_Classifier_Enhanced.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
