"""
Fine-tuning script to extend the Food Classifier with additional food categories.

Usage:
1. Create folders in data/raw/additional_foods/ (e.g., french_fries/, cola/, etc.)
2. Add 20-50 images to each folder
3. Run: python src/finetune_model.py

This script will:
- Load the pre-trained model
- Add new output classes
- Fine-tune on the new foods + existing Food-101 data
- Save the updated model
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import sys

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # Fine-tuning uses fewer epochs
AUTOTUNE = tf.data.AUTOTUNE

# Enable Mixed Precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def load_additional_foods():
    """Load images from additional_foods directory."""
    base_dir = Path(__file__).resolve().parent.parent
    additional_foods_dir = base_dir / "data" / "raw" / "additional_foods"
    
    image_paths = []
    labels = []
    class_to_idx = {}
    
    if not additional_foods_dir.exists():
        print(f"Creating {additional_foods_dir}")
        additional_foods_dir.mkdir(parents=True, exist_ok=True)
        return image_paths, labels, class_to_idx
    
    idx = 0
    for food_dir in sorted(additional_foods_dir.iterdir()):
        if food_dir.is_dir():
            class_name = food_dir.name
            class_to_idx[class_name] = idx
            
            image_count = 0
            for img_path in food_dir.glob("*.jpg"):
                image_paths.append(str(img_path))
                labels.append(class_name)
                image_count += 1
            
            if image_count > 0:
                print(f"  {class_name}: {image_count} images")
                idx += 1
    
    return image_paths, labels, class_to_idx

def create_food101_dataset(base_dir):
    """Load Food-101 dataset paths."""
    food101_dir = base_dir / "data" / "raw" / "food-101" / "images"
    
    if not food101_dir.exists():
        print("Food-101 dataset not found. Download it first.")
        return [], [], {}
    
    image_paths = []
    labels = []
    class_to_idx = {}
    
    idx = 0
    for class_dir in sorted(food101_dir.iterdir())[:50]:  # Use subset for faster fine-tuning
        if class_dir.is_dir():
            class_name = class_dir.name
            class_to_idx[class_name] = idx
            
            for img_path in list(class_dir.glob("*.jpg"))[:10]:  # Use 10 images per class
                image_paths.append(str(img_path))
                labels.append(class_name)
            
            idx += 1
    
    return image_paths, labels, class_to_idx

def load_and_preprocess_image(img_path, label_idx):
    """Load and preprocess an image."""
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label_idx

def get_augmenter():
    """Data augmentation for fine-tuning."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ])

def finetune_model(base_dir):
    """Fine-tune the model with additional foods."""
    
    print("\n" + "="*60)
    print("FOOD CLASSIFIER FINE-TUNING")
    print("="*60)
    
    # Load existing label map
    label_map_path = base_dir / "data" / "processed" / "label_map.json"
    with open(label_map_path, 'r') as f:
        existing_labels = json.load(f)
    
    max_idx = max(int(k) for k in existing_labels.keys())
    print(f"\nExisting classes: {len(existing_labels)}")
    
    # Load additional foods
    print("\nLoading additional food classes...")
    add_paths, add_labels, add_class_map = load_additional_foods()
    
    if not add_paths:
        print("ERROR: No additional food images found!")
        print(f"Create folders in: {base_dir / 'data' / 'raw' / 'additional_foods'}")
        print("Example: french_fries/, cola/, burger/")
        return False
    
    print(f"Additional classes found: {len(add_class_map)}")
    
    # Update label map with new classes
    updated_label_map = existing_labels.copy()
    for class_name, idx in add_class_map.items():
        new_idx = max_idx + idx + 1
        updated_label_map[str(new_idx)] = class_name
        print(f"  Added: {class_name} (index {new_idx})")
    
    # Load some Food-101 data for regularization (optional, skip if not available)
    print("\nLoading Food-101 subset for regularization...")
    f101_paths, f101_labels, f101_class_map = create_food101_dataset(base_dir)
    
    # If Food-101 not available, just use additional foods
    if not f101_paths:
        print("Food-101 not available, using only additional foods")
        f101_paths, f101_labels = [], []
    
    # Combine datasets
    all_paths = add_paths + f101_paths
    all_labels = add_labels + f101_labels
    
    # Create label to index mapping
    unique_classes = sorted(set(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
    
    label_indices = [label_to_idx[label] for label in all_labels]
    num_classes = len(unique_classes)
    
    print(f"\nTotal training samples: {len(all_paths)}")
    print(f"Total classes: {num_classes}")
    
    # Train-test split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, label_indices, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(
        lambda x, y: load_and_preprocess_image(x, y),
        num_parallel_calls=AUTOTUNE
    )
    
    augmenter = get_augmenter()
    def augment_fn(x, y):
        x = tf.cast(x, tf.float32)  # Ensure float32 for augmentation
        x = augmenter(x, training=True)
        return x, y
    
    train_ds = train_ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(
        lambda x, y: load_and_preprocess_image(x, y),
        num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    # Load pre-trained model
    print("\nLoading pre-trained classifier...")
    model_path = base_dir / "models" / "food_classifier_custom.h5"
    
    try:
        model = tf.keras.models.load_model(str(model_path))
        print(f"Loaded model with {model.layers[-1].units} output classes")
        
        # Create new model output layer for updated classes
        # Remove the last layer
        model_input = model.input
        x = model.layers[-2].output
        
        # Add new dense layer for new number of classes
        output = tf.keras.layers.Dense(num_classes, activation='softmax', dtype=tf.float32)(x)
        model = tf.keras.Model(inputs=model_input, outputs=output)
        
        # Compile with low learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Modified model to output {num_classes} classes")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Creating new model for training...")
        # Fallback: create a simple new model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax', dtype=tf.float32)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Train
    print("\nFine-tuning model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
    )
    
    # Save fine-tuned model
    print("\nSaving fine-tuned model...")
    model.save(str(model_path))
    print(f"Model saved to {model_path}")
    
    # Update label map
    with open(label_map_path, 'w') as f:
        json.dump(updated_label_map, f, indent=2)
    print(f"Label map updated with {len(updated_label_map)} classes")
    
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"New model can now detect: {', '.join(unique_classes[:10])}...")
    
    return True

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    
    # Check if there are additional foods to train on
    add_dir = base_dir / "data" / "raw" / "additional_foods"
    if not add_dir.exists() or len(list(add_dir.iterdir())) == 0:
        print("\n" + "!"*60)
        print("NO ADDITIONAL FOOD CLASSES FOUND")
        print("!"*60)
        print("\nTo fine-tune the model:")
        print(f"1. Create folders in: {add_dir}")
        print("   Example structure:")
        print("   additional_foods/")
        print("   ├── french_fries/")
        print("   │   ├── img1.jpg")
        print("   │   ├── img2.jpg")
        print("   │   └── ...")
        print("   └── cola/")
        print("       ├── img1.jpg")
        print("       └── ...")
        print("\n2. Add 20-50 images per food category")
        print("3. Run: python src/finetune_model.py")
        print("\nFor automated image download, run:")
        print("   python src/download_additional_foods.py")
        print("!"*60)
    else:
        success = finetune_model(base_dir)
        sys.exit(0 if success else 1)
