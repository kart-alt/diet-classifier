import os
import subprocess
import zipfile
import tarfile
import pandas as pd
from pathlib import Path
import json

def download_with_kaggle(dataset_str, download_dir):
    """Downloads a Kaggle dataset using the Kaggle API."""
    print(f"Downloading {dataset_str} via Kaggle API...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_str, "-p", str(download_dir)], check=True)
        # Find the downloaded zip and extract it
        zip_name = dataset_str.split("/")[-1] + ".zip"
        zip_path = download_dir / zip_name
        if zip_path.exists():
            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir / dataset_str.split("/")[-1])
            os.remove(zip_path) # Clean up zip
            print(f"Extraction of {dataset_str} complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dataset_str} via Kaggle: {e}")
        return False
    except FileNotFoundError:
        print("Kaggle CLI not found. Please ensure it is installed and in your PATH.")
        return False

def download_with_wget(url, download_dir, filename):
    """Downloads a file using wget fallback."""
    print(f"Downloading {url} via wget fallback...")
    out_path = download_dir / filename
    try:
        subprocess.run(["wget", "-O", str(out_path), url], check=True)
        print(f"Downloaded {filename}.")
        return out_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url} via wget: {e}")
        return None
    except FileNotFoundError:
         print("wget not found. Please ensure it is installed.")
         return None

def extract_tar_gz(tar_path, extract_path):
    """Extracts a tar.gz file."""
    print(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Extraction of {tar_path} complete.")
    except Exception as e:
         print(f"Failed to extract {tar_path}: {e}")

def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent
    data_raw_dir = base_dir / "data" / "raw"
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Datasets will be downloaded to: {data_raw_dir}")

    # Check for kaggle.json
    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    has_kaggle = kaggle_creds.exists()
    
    if not has_kaggle:
        print("WARNING: kaggle.json not found in ~/.kaggle/. Kaggle API downloads may fail unless environment variables are set.")
    
    # 1. Food-101
    dataset_101 = "dansbecker/food-101"
    if not download_with_kaggle(dataset_101, data_raw_dir):
        print("Falling back to wget for Food-101...")
        url_101 = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        tar_path = download_with_wget(url_101, data_raw_dir, "food-101.tar.gz")
        if tar_path and tar_path.exists():
            extract_tar_gz(tar_path, data_raw_dir)
            os.remove(tar_path) # Clean up tar
            
    # 2. USDA Nutritional Database
    print("Downloading USDA Nutritional Database...")
    usda_url = "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/nutrients.csv"
    try:
        usda_df = pd.read_csv(usda_url)
        usda_path = data_raw_dir / "usda_nutrients.csv"
        usda_df.to_csv(usda_path, index=False)
        print(f"Saved USDA nutrients to {usda_path}")
    except Exception as e:
        print(f"Failed to download USDA database: {e}")

    # 3. UEC Food-256
    dataset_uec = "imbikramsaha/uec-food-256"
    download_with_kaggle(dataset_uec, data_raw_dir)
    
    # 4. Open Food Facts
    dataset_off = "openfoodfacts/world-food-facts"
    download_with_kaggle(dataset_off, data_raw_dir)
    
    # 5. Nutrition5k (Supplementary for Portion size regression)
    # Using a common Kaggle repository for Nutrition5k if available, else a placeholder script
    # It is usually a large dataset from google research
    dataset_n5k = "trolukovich/nutrition5k-dataset" # One known Kaggle mirror of it
    if not download_with_kaggle(dataset_n5k, data_raw_dir):
         print("Nutrition5k mirror download failed. Skipping or requires manual gsutil download.")

    print("\n--- Download Summary ---")
    print(f"Files in {data_raw_dir}:")
    for item in data_raw_dir.iterdir():
        print(f"- {item.name}")
        
    print("\nCreating unified food_nutrition.csv...")
    # Attempt to create a naive unified nutrition CSV parsing USDA
    try:
        if (data_raw_dir / "usda_nutrients.csv").exists():
            df = pd.read_csv(data_raw_dir / "usda_nutrients.csv")
            # The CSV has columns Food, Measure, Grams, Calories, Protein, Fat, Sat.Fat, Fiber, Carbs
            # We standardize to per 100g
            df_cleaned = df.copy()
            # Clean numeric columns
            numeric_cols = ['Grams', 'Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']
            for col in numeric_cols:
                if col in df_cleaned.columns:
                     df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce').fillna(0)
                     
            # Convert to per 100g equivalent
            # Scale factor = 100 / Grams
            df_cleaned['Scale'] = 100 / df_cleaned['Grams'].replace(0, 1) # Avoid div by zero
            df_100g = pd.DataFrame()
            df_100g['Food'] = df_cleaned['Food']
            df_100g['Calories (kcal)'] = df_cleaned['Calories'] * df_cleaned['Scale']
            df_100g['Protein (g)'] = df_cleaned['Protein'] * df_cleaned['Scale']
            df_100g['Fat (g)'] = df_cleaned['Fat'] * df_cleaned['Scale']
            df_100g['Carbs (g)'] = df_cleaned['Carbs'] * df_cleaned['Scale']
            df_100g['Fiber (g)'] = df_cleaned['Fiber'] * df_cleaned['Scale']
            
            output_csv_path = data_raw_dir / "food_nutrition.csv"
            df_100g.to_csv(output_csv_path, index=False)
            print(f"Unified matching table created at {output_csv_path}")
            print(f"Total nutrition entries available: {len(df_100g)}")
        else:
            print("USDA nutrients file not found to create unified matching table.")
    except Exception as e:
        print(f"Error creating unified food_nutrition.csv: {e}")

if __name__ == "__main__":
    main()
