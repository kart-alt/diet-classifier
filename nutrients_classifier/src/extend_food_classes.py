"""
Script to extend Food-101 classifier with additional basic food items.
You can use this to collect/organize images for fine-tuning.

Steps:
1. Create folders in data/raw/additional_foods/ for new classes like:
   - french_fries/
   - cola/
   - soda/
   - milkshake/
   - burger/
   - etc.
2. Add your training images to each folder
3. Run: python src/extend_food_classes.py
4. Then run: python src/train_model.py (with updated settings)
"""

import os
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def extend_label_map(base_dir):
    """Extend label map with new food classes."""
    
    additional_foods_dir = base_dir / "data" / "raw" / "additional_foods"
    label_map_path = base_dir / "data" / "processed" / "label_map.json"
    
    # Load existing label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    # Get next available index
    next_idx = len(label_map)
    
    # Add new food classes
    if additional_foods_dir.exists():
        for food_dir in additional_foods_dir.iterdir():
            if food_dir.is_dir():
                class_name = food_dir.name
                if class_name not in label_map.values():
                    label_map[str(next_idx)] = class_name
                    next_idx += 1
                    print(f"Added class: {class_name} (index: {next_idx-1})")
    
    # Save updated label map
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"Updated label map saved with {len(label_map)} classes")
    return label_map

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    print("To extend your classifier:")
    print(f"1. Create folders in {base_dir / 'data' / 'raw' / 'additional_foods'}")
    print("   Example: french_fries/, cola/, burger/, etc.")
    print("2. Add training images to each folder")
    print("3. Run: python src/extend_food_classes.py")
    print("4. Run: python src/train_model.py")
    
    label_map = extend_label_map(base_dir)
