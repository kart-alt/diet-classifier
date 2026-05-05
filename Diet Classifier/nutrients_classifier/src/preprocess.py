import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 224
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

def get_image_paths_and_labels(data_raw_dir):
    """
    Scans the raw data directories (Food-101 and UEC-256), extracts class names,
    and returns image paths and their corresponding labels.
    """
    image_paths = []
    labels = []
    class_names = set()
    
    # 1. Food-101
    food101_dir = data_raw_dir / "food-101" / "images"
    if food101_dir.exists():
        for class_dir in food101_dir.iterdir():
            if class_dir.is_dir():
                class_names.add(class_dir.name)
                for img_path in class_dir.glob("*.jpg"):
                    image_paths.append(str(img_path))
                    labels.append(class_dir.name)
                    
    # 2. UEC Food-256 (if downloaded and extracted)
    uec_dir = data_raw_dir / "uec-food-256"
    if uec_dir.exists():
        category_file = uec_dir / "category.txt"
        if category_file.exists():
            uec_classes = {}
            with open(category_file, "r") as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        idx, name = parts[0], parts[1].replace(" ", "_").lower()
                        uec_classes[idx] = name
                        class_names.add(name)
            
            for idx, name in uec_classes.items():
                class_dir = uec_dir / str(idx)
                if class_dir.exists():
                    for img_path in class_dir.glob("*.jpg"):
                        image_paths.append(str(img_path))
                        labels.append(name)
                        
    # Sort class names for determinism
    ordered_classes = sorted(list(class_names))
    class_to_idx = {name: i for i, name in enumerate(ordered_classes)}
    
    numeric_labels = [class_to_idx[name] for name in labels]
    
    return image_paths, numeric_labels, ordered_classes

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_string, label):
    feature = {
        'image': _bytes_feature(image_string),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecords(image_paths, labels, out_file):
    print(f"Writing {len(image_paths)} images to {out_file}...")
    with tf.io.TFRecordWriter(str(out_file)) as writer:
        for img_path, label in zip(image_paths, labels):
            try:
                img_string = open(img_path, 'rb').read()
                # Verify parseable
                tf.io.decode_jpeg(img_string)
                example = serialize_example(img_string, int(label))
                writer.write(example)
            except Exception as e:
                # Corrupt image
                continue

def create_unified_exact_mapping(class_names, db_csv_path, out_json_path):
    """
    Creates an exact dictionary mapping for all 101/256 classes to the closest USDA DB match.
    Prevents fuzzy matching at runtime for O(1) deterministic access.
    """
    from fuzzywuzzy import process
    
    mapping = {}
    if Path(db_csv_path).exists():
        df = pd.read_csv(db_csv_path)
        db_foods = df['Food'].tolist() if 'Food' in df.columns else []
        db_dict = df.set_index('Food').to_dict(orient='index') if 'Food' in df.columns else {}
        
        for c_name in class_names:
            search_term = c_name.replace("_", " ")
            if db_foods:
                best_match, score = process.extractOne(search_term, db_foods)
                if score > 60:
                    nutrients = db_dict[best_match]
                    mapping[c_name] = {
                        "match_name": best_match,
                        "confidence": score,
                        "calories": nutrients.get("Calories (kcal)", 0),
                        "protein": nutrients.get("Protein (g)", 0),
                        "fat": nutrients.get("Fat (g)", 0),
                        "carbs": nutrients.get("Carbs (g)", 0)
                    }
                else:
                    mapping[c_name] = {"match_name": "Unknown", "calories": 0}
            else:
                 mapping[c_name] = {"match_name": "Unknown", "calories": 0}
    else:
        for c_name in class_names:
            mapping[c_name] = {"match_name": "No DB", "calories": 0}
            
    with open(out_json_path, 'w') as f:
        json.dump(mapping, f, indent=4)
        
def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_raw_dir = base_dir / "data" / "raw"
    data_proc_dir = base_dir / "data" / "processed"
    data_proc_dir.mkdir(parents=True, exist_ok=True)
    
    print("Gathering image paths and resolving classes...")
    image_paths, labels, class_names = get_image_paths_and_labels(data_raw_dir)
    num_classes = len(class_names)
    print(f"Total valid images found: {len(image_paths)}")
    print(f"Total distinct classes: {num_classes}")
    
    if len(image_paths) == 0:
        print("No images found. Please run download_datasets.py first or ensure paths are correct.")
        return

    # Save label map
    label_map = {i: name for i, name in enumerate(class_names)}
    with open(data_proc_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)
        
    print("Creating exact nutrition mapping dictionary...")
    create_unified_exact_mapping(
        class_names, 
        data_raw_dir / "food_nutrition.csv", 
        data_proc_dir / "nutrition_mapping.json"
    )

    # 70/15/15 Stratified Split
    print("Splitting datasets (70% train, 15% val, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Compute class weights for imbalanced datasets
    classes_unique = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)
    class_weights_dict = {int(k): float(v) for k, v in zip(classes_unique, weights)}
    
    with open(data_proc_dir / "class_weights.json", "w") as f:
        json.dump(class_weights_dict, f, indent=4)

    # Note: Full TFRecords for 130k images may take a lot of disk space.
    # To be extremely robust, we will output chunks or simply write them out.
    write_tfrecords(X_train, y_train, data_proc_dir / "train.tfrecord")
    write_tfrecords(X_val, y_val, data_proc_dir / "val.tfrecord")
    write_tfrecords(X_test, y_test, data_proc_dir / "test.tfrecord")
    print("Preprocessing completed. TFRecords saved to /data/processed/")

if __name__ == "__main__":
    main()
