"""
Automated food image downloader for fine-tuning.

Downloads images from Bing Image Search for specific food categories.
Requires: pip install bing-image-downloader

Usage:
    python src/download_additional_foods.py
    
This will download ~50 images each for:
    - french_fries
    - cola/soda
    - burger
    - pizza (additional)
    - milkshake
    - chips
    - ice_cream
"""

import os
import shutil
from pathlib import Path
import time
import subprocess
import sys

# Food categories to download with search queries
FOOD_CATEGORIES = {
    "french_fries": "crispy golden french fries",
    "cola": "cold cola soda drink",
    "burger": "delicious hamburger",
    "milkshake": "milkshake drink",
    "chips": "potato chips snack",
    "ice_cream": "ice cream dessert",
    "donut": "donut doughnut",
    "coffee": "coffee cup beverage",
}

IMAGES_PER_CATEGORY = 40  # Download 40 images per food type

def install_bing_downloader():
    """Install bing-image-downloader if not present."""
    try:
        import bing_image_downloader
        return True
    except ImportError:
        print("Installing bing-image-downloader...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bing-image-downloader"])
        return True

def download_images_bing(food_name, query, output_dir, count=30):
    """Download images from Bing Image Search."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {count} images for '{food_name}'...")
    
    try:
        from bing_image_downloader import downloader
        
        # Create temp directory for download
        temp_dir = Path("temp_bing_download")
        
        downloader.download(
            query,
            limit=count,
            output_dir=str(temp_dir),
            adult_filter_off=True,
            force_replace=False,
            timeout=15,
            verbose=False
        )
        
        # Move images to target directory
        temp_query_dir = temp_dir / query
        if temp_query_dir.exists():
            img_count = 0
            for img_path in temp_query_dir.glob("*.jpg"):
                try:
                    dest = output_dir / f"{food_name}_{img_count:03d}.jpg"
                    shutil.copy(str(img_path), str(dest))
                    img_count += 1
                except:
                    pass
            
            # Cleanup
            try:
                shutil.rmtree(temp_query_dir)
            except:
                pass
            
            if img_count > 0:
                print(f"✓ Downloaded {img_count} images to {output_dir}")
                return True
        
        return False
        
    except Exception as e:
        print(f"Error downloading for {food_name}: {e}")
        return False

def download_images_unsplash(food_name, query, output_dir, count=30):
    """Fallback: Show manual download instructions."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nManual download for '{food_name}':")
    print(f"  Search: {query}")
    print(f"  Save to: {output_dir}")
    
    return False

def setup_additional_foods():
    """Setup folder structure for additional foods."""
    
    base_dir = Path(__file__).resolve().parent.parent
    additional_foods_dir = base_dir / "data" / "raw" / "additional_foods"
    
    additional_foods_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("FOOD IMAGE DOWNLOADER FOR FINE-TUNING")
    print("="*60)
    
    print("\nInstalling dependencies...")
    install_bing_downloader()
    
    print("\nThis script will download training images for:")
    for food in FOOD_CATEGORIES.keys():
        print(f"  • {food}")
    
    print("\n" + "-"*60)
    print("Attempting to download images...")
    print("(This may take 5-10 minutes, please wait)")
    print("-"*60)
    
    success_count = 0
    failed_foods = []
    
    for food_name, query in FOOD_CATEGORIES.items():
        output_dir = additional_foods_dir / food_name
        
        # Try Bing downloader
        if download_images_bing(food_name, query, output_dir, IMAGES_PER_CATEGORY):
            success_count += 1
        else:
            failed_foods.append(food_name)
            # Create empty folder for manual addition
            output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successfully downloaded for: {success_count}/{len(FOOD_CATEGORIES)} categories")
    
    if failed_foods:
        print(f"\nCould not auto-download for: {', '.join(failed_foods)}")
        print("\nYou can manually add images to:")
        for food in failed_foods:
            folder = additional_foods_dir / food
            print(f"  {folder}")
        
        print("\n📌 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("-"*60)
        for food in failed_foods[:3]:  # Show first 3
            print(f"\n{food}:")
            query = FOOD_CATEGORIES[food]
            print(f"  1. Go to: https://www.google.com/search?q=images+{query.replace(' ', '+')}")
            print(f"  2. Right-click → 'Save image as'")
            print(f"  3. Save to: {additional_foods_dir / food}")
            print(f"  4. Repeat for 20-30 images")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Review downloaded/added images:")
    print(f"   {additional_foods_dir}")
    print("\n2. Delete low-quality images (optional)")
    print("\n3. Run fine-tuning:")
    print("   python src/finetune_model.py")
    print("="*60)

if __name__ == "__main__":
    setup_additional_foods()
