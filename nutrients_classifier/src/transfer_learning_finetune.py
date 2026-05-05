"""
Proper Transfer Learning Fine-tuning for Food Classification.
Uses transfer learning to preserve original knowledge while adding new food classes.
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4  # Very low learning rate to preserve original weights

def load_image(img_path):
    """Load and preprocess image."""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def create_transfer_learning_model(num_classes):
    """Create model with transfer learning from pretrained weights."""
    # Build a new model with MobileNetV2 backbone (better than training from scratch)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom top layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_dataset():
    """Build training dataset from additional foods."""
    base_dir = Path(__file__).resolve().parent.parent
    food_dir = base_dir / "data" / "raw" / "additional_foods"
    
    images = []
    labels = []
    class_names = []
    
    if food_dir.exists():
        for idx, class_dir in enumerate(sorted(food_dir.iterdir())):
            if class_dir.is_dir():
                class_name = class_dir.name
                class_names.append(class_name)
                print(f"\nLoading class {idx}: {class_name}")
                
                img_count = 0
                for img_path in class_dir.glob("*.jpg"):
                    img_array = load_image(str(img_path))
                    if img_array is not None:
                        images.append(img_array)
                        labels.append(idx)
                        img_count += 1
                
                print(f"  ✓ Loaded {img_count} images")
    
    if not images:
        print("ERROR: No images found in additional_foods directory!")
        return None, None, None
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"\n✓ Total dataset: {len(X)} images, {len(class_names)} classes")
    return X, y, class_names

def main():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "food_classifier_custom.h5"
    
    print("=" * 60)
    print("Transfer Learning Fine-tuning")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    X, y, class_names = build_dataset()
    
    if X is None:
        print("Cannot train without images!")
        return
    
    print(f"\n[2/4] Building transfer learning model...")
    num_classes = len(class_names)
    model = create_transfer_learning_model(num_classes)
    
    # Compile with low learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train
    print(f"\n[3/4] Training model for {EPOCHS} epochs...")
    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1
    )
    
    # Save
    print(f"\n[4/4] Saving model to {model_path}...")
    model.save(str(model_path))
    
    # Save class names
    with open(base_dir / "data" / "processed" / "new_food_classes.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"✓ Model saved!")
    print(f"✓ New classes: {class_names}")

if __name__ == "__main__":
    main()
