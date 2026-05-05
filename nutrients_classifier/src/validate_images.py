"""
Image validation and conversion script.
Converts all downloaded images to a standard format to ensure training compatibility.
"""

from PIL import Image
from pathlib import Path
import sys

IMG_SIZE = 224

def validate_and_convert_images(base_dir):
    """Validate and convert all images to RGB JPEG."""
    
    additional_foods_dir = base_dir / "data" / "raw" / "additional_foods"
    
    if not additional_foods_dir.exists():
        print(f"ERROR: {additional_foods_dir} not found")
        return False
    
    print("\n" + "="*60)
    print("IMAGE VALIDATION & CONVERSION")
    print("="*60)
    
    total_processed = 0
    total_converted = 0
    total_deleted = 0
    
    for food_dir in sorted(additional_foods_dir.iterdir()):
        if not food_dir.is_dir():
            continue
        
        print(f"\nProcessing {food_dir.name}...")
        converted = 0
        deleted = 0
        
        for img_path in list(food_dir.glob("*")):
            if not img_path.is_file():
                continue
            
            total_processed += 1
            
            try:
                # Open image
                with Image.open(img_path) as img:
                    # Convert to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Check minimum size (at least 100x100)
                    if img.width < 100 or img.height < 100:
                        print(f"  ✗ Deleting {img_path.name} (too small: {img.width}x{img.height})")
                        img_path.unlink()
                        total_deleted += 1
                        deleted += 1
                        continue
                    
                    # Save as JPEG
                    new_path = img_path.parent / f"{img_path.stem}.jpg"
                    img.save(new_path, 'JPEG', quality=95)
                    
                    # Delete original if different
                    if new_path != img_path and img_path.exists():
                        img_path.unlink()
                    
                    converted += 1
                    total_converted += 1
                    
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
                try:
                    img_path.unlink()
                    total_deleted += 1
                    deleted += 1
                except:
                    pass
        
        if converted > 0:
            print(f"  ✓ Converted: {converted}, Deleted: {deleted}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"Total images processed: {total_processed}")
    print(f"Total converted: {total_converted}")
    print(f"Total deleted: {total_deleted}")
    print(f"Good images available: {total_converted}")
    
    return total_converted > 0

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    success = validate_and_convert_images(base_dir)
    print("\nRun training next: python src/train_simple.py")
    sys.exit(0 if success else 1)
