import os
import zipfile
from pathlib import Path

def download_dataset():
    # Base directory for the project
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "detection_dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # We will use Kaggle API to download the dataset.
    # The user must have kaggle.json configured in ~/.kaggle/kaggle.json
    
    # Example dataset: "sainikhilesh/food-object-detection" or similar
    dataset_name = "sainikhilesh/food-object-detection"
    
    print(f"Downloading {dataset_name} to {data_dir}...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=str(data_dir), unzip=True)
        print(f"✅ Successfully downloaded and unzipped the dataset to {data_dir}")
        print("You can now use this dataset to train a custom YOLO model.")
    except Exception as e:
        print(f"⚠️ Error downloading dataset: {e}")
        print("Please make sure you have your Kaggle API credentials set up correctly.")
        print("1. Create a Kaggle account.")
        print("2. Go to your account settings and generate a new API token (kaggle.json).")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows).")

if __name__ == "__main__":
    download_dataset()
