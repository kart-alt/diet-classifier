import os
import zipfile
import json
import shutil
import pathlib

# ==========================================
# AUTO-CONFIGURE KAGGLE TOKEN
# Searches for kaggle.json in Downloads or .kaggle folder
# and sets it up automatically.
# ==========================================
def setup_kaggle_credentials():
    kaggle_dir = pathlib.Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if kaggle_json.exists():
        print(f"Kaggle token found at: {kaggle_json}")
        return True

    # Search common download locations
    search_paths = [
        pathlib.Path.home() / 'Downloads',
        pathlib.Path.home() / 'Desktop',
        pathlib.Path('.'),
    ]
    found = None
    for folder in search_paths:
        for candidate in folder.glob('kaggle*.json'):
            found = candidate
            break
        if found:
            break

    if found:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(found), str(kaggle_json))
        # Kaggle API requires restrictive permissions on Unix; skip on Windows
        try:
            kaggle_json.chmod(0o600)
        except Exception:
            pass
        print(f"Auto-configured Kaggle token from: {found}")
        return True
    else:
        print("ERROR: Could not find kaggle.json anywhere.")
        print("Please download it from https://www.kaggle.com/settings (API section) and place it in:")
        print(f"  {kaggle_json}")
        return False

if not setup_kaggle_credentials():
    exit(1)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("The 'kaggle' module is not installed. Please run: pip install kaggle")
    exit(1)

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1. Dataset Configuration
# ==========================================
DATASET_SLUG = "iamsouravbanerjee/indian-food-images-dataset"
ZIP_FILE = "indian-food-images-dataset.zip"
EXTRACT_DIR = "data/indian_food_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 # Increase to 30+ for production accuracy

def download_and_extract_dataset():
    if not os.path.exists(EXTRACT_DIR):
        print(f"Connecting to Kaggle to download: {DATASET_SLUG}...")
        api = KaggleApi()
        api.authenticate()
        
        # Download
        api.dataset_download_files(DATASET_SLUG, path=".", unzip=False)
        print("Download complete. Extracting dataset...")
        
        # Extract
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete.")
        
        # Clean up zip
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
    else:
        print(f"Dataset already exists at {EXTRACT_DIR}. Skipping download.")

# ==========================================
# 2. Setup Data Generators
# ==========================================
def setup_data_generators(data_dir):
    # Some datasets have an inner folder. Let's find the actual directory with class folders.
    # The Indian Food Dataset from Kaggle usually extracts to `data/indian_food_dataset/Indian Food Images/Indian Food Images`
    
    actual_data_dir = data_dir
    for root, dirs, files in os.walk(data_dir):
        if len(dirs) > 10: # Found the directory with all the class folders
            actual_data_dir = root
            break
            
    print(f"Loading images from: {actual_data_dir}")

    # Create ImageDataGenerator with data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # 80% training, 20% validation
    )

    train_generator = datagen.flow_from_directory(
        actual_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        actual_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

# ==========================================
# 3. Build Transfer Learning Model
# ==========================================
def build_model(num_classes):
    print(f"Building MobileNetV2 model for {num_classes} classes...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# 4. Main Training Pipeline
# ==========================================
def main():
    print("--- Starting Automated Kaggle Training Pipeline ---")
    download_and_extract_dataset()
    
    train_gen, val_gen = setup_data_generators(EXTRACT_DIR)
    
    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    
    # Save the label map
    label_map = {str(v): k for k, v in class_indices.items()}
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/indian_label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map with {num_classes} classes to 'data/processed/indian_label_map.json'")
    
    model = build_model(num_classes)
    
    print(f"Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )
    
    # Save the final model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/indian_food_model.h5'
    model.save(model_path)
    print(f"Training Complete! Model saved successfully to '{model_path}'.")
    print("To use this model, update the 'api.py' to load 'indian_food_model.h5' and 'indian_label_map.json'.")

if __name__ == "__main__":
    main()
