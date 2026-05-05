"""
Simplified fine-tuning script that trains from scratch with the new food categories.
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30  # Fewer epochs for quick training
AUTOTUNE = tf.data.AUTOTUNE

def load_images_from_folders(base_dir):
    """Load all images from additional_foods folders."""
    
    additional_foods_dir = base_dir / "data" / "raw" / "additional_foods"
    
    image_paths = []
    labels = []
    class_names = []
    label_to_idx = {}
    
    if not additional_foods_dir.exists():
        print(f"ERROR: {additional_foods_dir} not found")
        return [], [], {}, []
    
    idx = 0
    for food_dir in sorted(additional_foods_dir.iterdir()):
        if food_dir.is_dir():
            class_name = food_dir.name
            class_names.append(class_name)
            label_to_idx[class_name] = idx
            
            img_count = 0
            invalid_count = 0
            for img_path in food_dir.glob("*.jpg"):
                try:
                    # Quick validation: try to open the image
                    from PIL import Image
                    try:
                        with Image.open(img_path) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            # Only add if valid
                            image_paths.append(str(img_path))
                            labels.append(idx)
                            img_count += 1
                    except:
                        invalid_count += 1
                except:
                    invalid_count += 1
            
            if img_count > 0:
                print(f"  {class_name}: {img_count} valid, {invalid_count} invalid")
                idx += 1
    
    print(f"\nTotal valid images: {len(image_paths)}")
    print(f"Total classes: {len(class_names)}")
    
    return image_paths, labels, label_to_idx, class_names

def load_and_preprocess_image(img_path, label_idx):
    """Load and preprocess an image."""
    image = tf.io.read_file(img_path)
    
    # Try to decode as JPEG, fallback to PNG
    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except:
        try:
            image = tf.image.decode_png(image, channels=3)
        except:
            # Fallback: return a valid blank image
            return tf.ones((IMG_SIZE, IMG_SIZE, 3)), label_idx
    
    # Ensure it's RGB (3 channels)
    if len(image.shape) < 3:
        image = tf.expand_dims(image, -1)
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 4:
        image = image[:,:,:3]  # Remove alpha channel
    
    # Ensure uint8 before resize
    image = tf.cast(image, tf.uint8)
    
    # Resize
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalize
    image = image / 255.0
    
    return image, label_idx

def build_model(num_classes):
    """Build a simple CNN for food classification."""
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Classification head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    """Train the food classifier."""
    
    base_dir = Path(__file__).resolve().parent.parent
    
    print("\n" + "="*60)
    print("FOOD CLASSIFIER TRAINING")
    print("="*60)
    
    # Load images
    print("\nLoading training images...")
    image_paths, labels, label_to_idx, class_names = load_images_from_folders(base_dir)
    
    if not image_paths:
        print("ERROR: No images found!")
        return False
    
    # Convert labels to numpy array
    labels = np.array(labels)
    num_classes = len(class_names)
    
    # Train-test split
    print("\nSplitting data (80% train, 20% val)...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    
    # Create datasets
    print("\nPreparing datasets...")
    
    def make_dataset(paths, labels, augment=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.shuffle(len(paths)) if augment else ds
        
        ds = ds.map(
            lambda x, y: load_and_preprocess_image(x, y),
            num_parallel_calls=AUTOTUNE
        )
        
        if augment:
            def augment_fn(x, y):
                x = tf.image.random_flip_left_right(x)
                x = tf.image.random_brightness(x, 0.1)
                x = tf.image.random_contrast(x, 0.9, 1.1)
                return x, y
            
            ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
        
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    
    train_ds = make_dataset(train_paths, train_labels, augment=True)
    val_ds = make_dataset(val_paths, val_labels, augment=False)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(num_classes)
    print(model.summary())
    
    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]
    )
    
    # Save model
    print("\nSaving model...")
    model_path = base_dir / "models" / "food_classifier_custom.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
    # Save label map
    print("\nUpdating label map...")
    label_map_path = base_dir / "data" / "processed" / "label_map.json"
    label_map = {str(i): name for i, name in enumerate(class_names)}
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"✓ Label map saved with {len(label_map)} classes")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model can now detect: {', '.join(class_names)}")
    print(f"Test it: python app.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = train()
    sys.exit(0 if success else 1)
