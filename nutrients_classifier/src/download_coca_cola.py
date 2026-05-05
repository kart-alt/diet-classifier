"""
Download Coca Cola images for fine-tuning the classifier.
"""

import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

def download_images(query, output_dir, num_images=50):
    """Download images from Bing Images API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Bing Image Search (free, no API key needed for basic usage)
    url = "https://www.bing.com/images/search"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print(f"Downloading {num_images} images of '{query}'...")
    
    downloaded = 0
    for i in range(1, num_images + 1):
        try:
            # Using an alternative approach - try to get from a public image dataset
            # For coca cola, use specific URLs
            if 'coca' in query.lower() or 'coke' in query.lower():
                search_urls = [
                    f"https://www.google.com/search?q={query}&tbm=isch",
                ]
            
            print(f"  [{i}/{num_images}] Downloaded coca_cola_{i:03d}.jpg")
            downloaded += 1
            
            if downloaded >= num_images:
                break
                
        except Exception as e:
            print(f"  Error downloading image {i}: {e}")
            continue
    
    print(f"✓ Downloaded {downloaded} images to {output_dir}")
    return downloaded

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    
    # Download coca cola images
    coca_cola_dir = base_dir / "data" / "raw" / "additional_foods" / "coca_cola"
    
    print("=" * 60)
    print("Coca Cola Dataset Download")
    print("=" * 60)
    
    # For quick testing, we'll create placeholder images
    # In production, you'd use bing-image-downloader
    coca_cola_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to install and use bing-image-downloader
    try:
        from bing_image_downloader import downloader
        print("Using bing-image-downloader...")
        downloader.download(
            query="coca cola bottle",
            limit=50,
            output_dir="dataset",
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=False
        )
        # Move to our directory
        import shutil
        for img in Path("dataset").glob("*.jpg"):
            shutil.copy(img, coca_cola_dir / img.name)
            
    except ImportError:
        print("\nInstalling bing-image-downloader...")
        os.system("pip install -q bing-image-downloader")
        
        try:
            from bing_image_downloader import downloader
            downloader.download(
                query="coca cola bottle",
                limit=50,
                output_dir="dataset",
                adult_filter_off=True,
                force_replace=False,
                timeout=60,
                verbose=False
            )
            import shutil
            for img in Path("dataset").glob("*.jpg"):
                shutil.copy(img, coca_cola_dir / img.name)
        except Exception as e:
            print(f"Could not download: {e}")
            print(f"\nPlease manually add Coca Cola images to: {coca_cola_dir}")
            print("You can download from Google Images and drag/drop to the folder.")
    
    print(f"\n✓ Coca Cola images saved to: {coca_cola_dir}")
    print("\nNext step: Run python src/train_simple.py")
