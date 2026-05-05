import os
import sys
from PIL import Image

# Recommended env vars for verbose output
os.environ.setdefault('DEBUG_DETAILED', '1')
os.environ.setdefault('FORCE_ACCEPT_DETECTIONS', '1')
os.environ.setdefault('MIN_CLASS_CONF', '5')

# Importing app will load models and globals
from app import process_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_run.py <image_path>")
        return
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Image not found: {path}")
        return
    img = Image.open(path).convert('RGB')
    annotated_img, nutrient_html, macro_img, cal_img, pdf_path = process_image(img, 'English')
    print(f"Annotated image returned: {annotated_img is not None}")
    print(f"PDF path: {pdf_path}")
    # Save outputs for inspection
    out_dir = os.path.join(os.getcwd(), 'nutrients_classifier', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    if annotated_img is not None:
        annotated_img.save(os.path.join(out_dir, 'debug_annotated.png'))
        print('Saved annotated image to outputs/debug_annotated.png')
    try:
        if macro_img is not None:
            macro_img.save(os.path.join(out_dir, 'debug_macro.png'))
            print('Saved macro chart to outputs/debug_macro.png')
        if cal_img is not None:
            cal_img.save(os.path.join(out_dir, 'debug_cal.png'))
            print('Saved calorie chart to outputs/debug_cal.png')
    except Exception as e:
        print('Could not save charts:', e)

if __name__ == '__main__':
    main()
