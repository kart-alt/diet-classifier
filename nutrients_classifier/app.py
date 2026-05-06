import os
import json
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import datetime

from src.nutrition_engine import NutritionEngine
from src.utils import translate_to_tamil, plot_macronutrients, plot_calorie_distribution, generate_pdf_report

# ─── Global State ─────────────────────────────────────────────────────────────
print("Initializing App State and Loading Models...")
base_dir = Path(__file__).resolve().parent
classifier_path   = base_dir / "models" / "food_classifier_custom.h5"
portion_path      = base_dir / "models" / "portion_estimator.h5"
label_map_path    = base_dir / "data" / "processed" / "label_map.json"
yolo_mapping_path = base_dir / "data" / "processed" / "yolo_to_nutrition_mapping.json"

engine = NutritionEngine(base_dir / "data")

yolo_to_food_mapping = {}
try:
    with open(yolo_mapping_path) as f:
        yolo_to_food_mapping = json.load(f)
except Exception as e:
    print(f"Could not load YOLO mapping: {e}")

DEBUG_DETAILED          = os.environ.get("DEBUG_DETAILED", "0") == "1"
FORCE_ACCEPT_DETECTIONS = os.environ.get("FORCE_ACCEPT_DETECTIONS", "0") == "1"
MIN_CLASS_CONF          = float(os.environ.get("MIN_CLASS_CONF", "0.01"))
YOLO_MIN_CONF           = float(os.environ.get("YOLO_MIN_CONF", "0.05"))

try:
    classifier = tf.keras.models.load_model(str(classifier_path))
    print("Custom Classifier loaded.")
except Exception as e:
    print(f"Classifier not found: {e}.")
    classifier = None

try:
    portion_estimator = tf.keras.models.load_model(str(portion_path))
    print("Portion Estimator loaded.")
except Exception:
    portion_estimator = None

try:
    with open(label_map_path) as f:
        raw_map = json.load(f)
    # Remove the placeholder entry "101": "Food Classification" if present
    label_map = {k: v for k, v in raw_map.items() if v != "Food Classification"}
    print(f"Label map loaded: {len(label_map)} classes.")
except Exception:
    label_map = {str(i): f"Food_{i}" for i in range(101)}

try:
    from ultralytics import YOLO
    object_detector = YOLO(str(base_dir / "yolov8n.pt"))
    print("YOLOv8 model loaded.")
except Exception as e:
    print(f"Could not load YOLO: {e}.")
    object_detector = None

IMG_SIZE = 224

# ─── Meal Log State ────────────────────────────────────────────────────────────
meal_log = []  # list of dicts: {food, grams, meal_type, time, nutrients}

# ─── DRI / Meta ───────────────────────────────────────────────────────────────
DRI = {
    "Calories (kcal)": 2000, "Protein (g)": 50, "Fat (g)": 70,
    "Carbs (g)": 300, "Fiber (g)": 28, "Sugar (g)": 50,
    "Sodium (mg)": 2300, "Vitamin C (mg)": 90, "Calcium (mg)": 1000, "Iron (mg)": 18,
}
NUTRIENT_META = {
    "Calories (kcal)": ("🔥 Calories",      "kcal", "var(--cal)", (300, 700)),
    "Protein (g)":     ("💪 Protein",       "g",    "var(--protein)", (10,  25)),
    "Fat (g)":         ("🥑 Fat",           "g",    "var(--fat)", (10,  25)),
    "Carbs (g)":       ("🍞 Carbohydrates", "g",    "var(--carbs)", (30,  80)),
    "Fiber (g)":       ("🌿 Fiber",         "g",    "var(--fiber)", (5,   15)),
    "Sugar (g)":       ("🍬 Sugar",         "g",    "var(--accent2)", (5,   20)),
    "Sodium (mg)":     ("🧂 Sodium",        "mg",   "var(--text-muted)", (300, 800)),
    "Vitamin C (mg)":  ("🍊 Vitamin C",     "mg",   "var(--accent2)", (15,  60)),
    "Calcium (mg)":    ("🦴 Calcium",       "mg",   "var(--accent3)", (200, 600)),
    "Iron (mg)":       ("⚙️ Iron",          "mg",   "var(--text-muted)", (3,   10)),
}

FOOD_TIPS = {
    "burger": "🍔 High in calories & sodium. Enjoy occasionally and pair with a salad.",
    "chips": "🥔 High in sodium & fat. Opt for baked versions or veggie chips.",
    "coffee": "☕ Moderate caffeine is fine. Avoid adding excessive sugar/cream.",
    "cola": "🥤 High in sugar. Try sparkling water with fruit for a healthier alternative.",
    "donut": "🍩 High in sugar & refined carbs. A treat for special occasions.",
    "french_fries": "🍟 High in fat & sodium. Prefer oven-baked sweet potato fries.",
    "ice_cream": "🍦 High in sugar & fat. Try frozen yogurt or fruit sorbet.",
    "milkshake": "🥛 Rich in calcium but high in sugar. Go for smoothies instead.",
}

HEALTHY_ALTERNATIVES = {
    "burger": ["Grilled chicken wrap", "Veggie patty burger", "Lettuce-wrapped burger"],
    "chips": ["Rice cakes", "Carrot sticks with hummus", "Air-popped popcorn"],
    "cola": ["Sparkling water with lemon", "Herbal iced tea", "Fresh fruit juice"],
    "donut": ["Whole grain muffin", "Greek yogurt with honey", "Fresh fruit salad"],
    "french_fries": ["Baked sweet potato fries", "Roasted zucchini sticks", "Cucumber slices"],
    "ice_cream": ["Frozen banana nice cream", "Greek yogurt parfait", "Fruit sorbet"],
    "milkshake": ["Protein smoothie", "Banana oat shake", "Green smoothie"],
}

MEAL_PLANS = {
    "Weight Loss": {
        "Breakfast": ["Oatmeal with berries (300 kcal)", "Greek yogurt parfait (250 kcal)", "Veggie egg white omelette (280 kcal)"],
        "Lunch":     ["Grilled chicken salad (400 kcal)", "Quinoa Buddha bowl (450 kcal)", "Turkey lettuce wraps (350 kcal)"],
        "Snack":     ["Apple with almond butter (200 kcal)", "Handful of mixed nuts (180 kcal)", "Celery with hummus (120 kcal)"],
        "Dinner":    ["Baked salmon with veggies (450 kcal)", "Stir-fry tofu with brown rice (500 kcal)", "Lentil soup (380 kcal)"],
    },
    "Muscle Gain": {
        "Breakfast": ["Protein pancakes with eggs (600 kcal)", "Oats with protein powder (550 kcal)", "Scrambled eggs with toast (500 kcal)"],
        "Lunch":     ["Chicken rice bowl (700 kcal)", "Beef & sweet potato (750 kcal)", "Tuna pasta salad (650 kcal)"],
        "Snack":     ["Protein shake with banana (400 kcal)", "Cottage cheese with nuts (350 kcal)", "Boiled eggs (200 kcal)"],
        "Dinner":    ["Steak with quinoa (800 kcal)", "Grilled chicken with pasta (750 kcal)", "Salmon with lentils (700 kcal)"],
    },
    "Balanced Health": {
        "Breakfast": ["Smoothie bowl (350 kcal)", "Whole grain toast with avocado (300 kcal)", "Muesli with milk (320 kcal)"],
        "Lunch":     ["Mediterranean wrap (500 kcal)", "Vegetable curry with rice (520 kcal)", "Chicken soup with bread (450 kcal)"],
        "Snack":     ["Mixed fruit (150 kcal)", "Trail mix (220 kcal)", "Low-fat cheese & crackers (200 kcal)"],
        "Dinner":    ["Grilled fish with veggies (500 kcal)", "Chickpea stew (480 kcal)", "Vegetable pasta (520 kcal)"],
    },
}

# ─── Helper functions ──────────────────────────────────────────────────────────
def _level_bar(value, max_val, color):
    pct = min(100, int((value / max_val) * 100)) if max_val else 0
    return (
        f'<div class="progress-track">'
        f'<div class="progress-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
    )

def _nutrient_level_label(value, thresholds):
    if value <= thresholds[0]:
        return "Low", "var(--accent3)"
    elif value <= thresholds[1]:
        return "Med", "var(--accent2)"
    else:
        return "High", "var(--fat)"

def get_emoji_for_food(food_name):
    f = food_name.lower()
    if 'burger' in f: return '🍔'
    if 'pizza' in f: return '🍕'
    if 'chicken' in f: return '🍗'
    if 'salad' in f: return '🥗'
    if 'rice' in f: return '🍚'
    if 'egg' in f: return '🍳'
    if 'coffee' in f: return '☕'
    if 'apple' in f: return '🍎'
    return '🍽️'

def build_nutrient_html(nutrients, detections_list):
    preds_html = ""
    for d in detections_list:
        try:
            cals = (d['grams'] / 100.0) * engine.get_nutrition_per_100g(d['food'].lower()).get("Calories (kcal)", 0)
        except Exception:
            cals = 0
        emoji = get_emoji_for_food(d['food'])
        preds_html += f"""
        <div class="detected-item fade-in-scroll">
          <div class="detected-emoji">{emoji}</div>
          <div class="detected-info">
            <div class="detected-name">{d['food']}</div>
            <div class="detected-meta">{d['grams']:.0f}g &bull; {cals:.0f} kcal</div>
          </div>
          <div class="detected-conf">
            <div class="conf-lbl">{d['conf']:.1f}% Match</div>
            <div class="progress-track"><div class="progress-fill" style="width:{d['conf']}%;background:var(--accent);"></div></div>
          </div>
        </div>"""

    p = nutrients.get("Protein (g)", 0)
    c = nutrients.get("Carbs (g)", 0)
    f = nutrients.get("Fat (g)", 0)
    fib = nutrients.get("Fiber (g)", 0)
    sod = nutrients.get("Sodium (mg)", 0)
    total_cals = nutrients.get("Calories (kcal)", 0)
    
    total_m = p + c + f if (p+c+f) > 0 else 1
    p_pct, c_pct, f_pct = (p/total_m)*100, (c/total_m)*100, (f/total_m)*100

    macros_html = f"""
    <div class="macro-summary">
        <div class="macro-total">
            <span class="macro-total-val">{total_cals:.0f}</span>
            <span class="macro-total-lbl">kcal</span>
        </div>
        <div class="macro-segmented-bar">
            <div style="width:{p_pct}%; background:var(--protein);" title="Protein"></div>
            <div style="width:{c_pct}%; background:var(--carbs);" title="Carbs"></div>
            <div style="width:{f_pct}%; background:var(--fat);" title="Fat"></div>
        </div>
        <div class="macro-cards">
            <div class="m-card" style="border-bottom: 3px solid var(--protein);">
                <div class="m-lbl">Protein</div><div class="m-val">{p:.1f}g</div>
            </div>
            <div class="m-card" style="border-bottom: 3px solid var(--carbs);">
                <div class="m-lbl">Carbs</div><div class="m-val">{c:.1f}g</div>
            </div>
            <div class="m-card" style="border-bottom: 3px solid var(--fat);">
                <div class="m-lbl">Fat</div><div class="m-val">{f:.1f}g</div>
            </div>
            <div class="m-card" style="border-bottom: 3px solid var(--fiber);">
                <div class="m-lbl">Fiber</div><div class="m-val">{fib:.1f}g</div>
            </div>
            <div class="m-card" style="border-bottom: 3px solid var(--text-muted);">
                <div class="m-lbl">Sodium</div><div class="m-val">{sod:.0f}mg</div>
            </div>
        </div>
    </div>
    """

    vits = ["Vitamin C (mg)", "Calcium (mg)", "Iron (mg)", "Sugar (g)"]
    vitamins_html = "<div class='vitamins-list'>"
    for v in vits:
        val = nutrients.get(v, 0)
        lbl, unit, col, thresh = NUTRIENT_META[v]
        pct = min(100, round((val / DRI[v]) * 100))
        vitamins_html += f"""
        <div class="vit-row">
            <div class="vit-lbl">{lbl}</div>
            <div class="vit-bar"><div class="progress-track"><div class="progress-fill" style="width:{pct}%;background:{col};"></div></div></div>
            <div class="vit-val">{val:.1f}{unit}</div>
        </div>"""
    vitamins_html += "</div>"

    insights_html = "<div class='insights-list'>"
    shown_foods = set()
    for d in detections_list:
        food_key = d["food"].lower().replace(" ", "_")
        if food_key in FOOD_TIPS and food_key not in shown_foods:
            shown_foods.add(food_key)
            insights_html += f"""
            <div class="insight-card">
                <div class="insight-icon">💡</div>
                <div class="insight-text">
                    <div class="insight-title">Tip for {d['food']}</div>
                    <div class="insight-desc">{FOOD_TIPS[food_key]}</div>
                </div>
            </div>"""
            if food_key in HEALTHY_ALTERNATIVES:
                alts = ", ".join(HEALTHY_ALTERNATIVES[food_key])
                insights_html += f"""
                <div class="insight-card" style="border-left-color: var(--accent);">
                    <div class="insight-icon">🥦</div>
                    <div class="insight-text">
                        <div class="insight-title">Healthier Alternatives</div>
                        <div class="insight-desc">{alts}</div>
                    </div>
                </div>"""
    if not shown_foods:
        insights_html += "<div class='insight-card'><div class='insight-text'><div class='insight-desc'>No specific insights for these items.</div></div></div>"
    insights_html += "</div>"

    dv_rows = ""
    for key, (label, unit, color, thresholds) in NUTRIENT_META.items():
        val = nutrients.get(key, 0)
        dri = DRI.get(key, 1)
        pct_dri = min(100, round((val / dri) * 100))
        dv_rows += f"""
        <div class="dv-row">
            <div class="dv-lbl">{label}</div>
            <div class="dv-bar"><div class="progress-track"><div class="progress-fill" style="width:{pct_dri}%;background:{color};"></div></div></div>
            <div class="dv-pct">{pct_dri}%</div>
        </div>"""

    return f"""
<div class="analysis-grid">
    <div class="results-col">
        <div class="surface-card fade-in-scroll">
            <h3 class="syne-header" style="font-size:1.5rem;margin-bottom:15px;color:var(--text);">Detected Foods</h3>
            {preds_html if preds_html else '<p style="color:var(--text-muted);">No food items detected.</p>'}
        </div>
        
        <div class="surface-card fade-in-scroll" style="margin-top:20px; padding:0;">
            <div class="css-tabs">
                <input type="radio" name="ntabs" id="tab-macros" checked>
                <label for="tab-macros">Macros</label>
                <input type="radio" name="ntabs" id="tab-vitamins">
                <label for="tab-vitamins">Micros</label>
                <input type="radio" name="ntabs" id="tab-insights">
                <label for="tab-insights">Insights</label>
                
                <div class="tab-content" id="content-macros">
                    {macros_html}
                </div>
                <div class="tab-content" id="content-vitamins">
                    {vitamins_html}
                </div>
                <div class="tab-content" id="content-insights">
                    {insights_html}
                </div>
            </div>
        </div>
    </div>
    
    <div class="sidebar-col">
        <div class="surface-card fade-in-scroll">
            <h3 class="syne-header" style="font-size:1.2rem;margin-bottom:15px;color:var(--text);">% Daily Value</h3>
            <div class="dv-list">
                {dv_rows}
            </div>
        </div>
    </div>
</div>"""

def classify_crop(crop_img, area_ratio=None):
    """Classify a cropped food image. Returns (food_name, conf_percent, est_grams)."""
    img_resized = crop_img.resize((IMG_SIZE, IMG_SIZE))
    img_array   = np.array(img_resized) / 255.0
    img_tensor  = np.expand_dims(img_array, axis=0)

    top_predictions = []

    if classifier is not None:
        preds = classifier.predict(img_tensor, verbose=0)[0]
        max_prob = float(preds.max())

        if max_prob < 0.02:
            food_name = "Unknown"
            conf      = max_prob * 100
        else:
            top_idxs = preds.argsort()[::-1][:3]
            for idx in top_idxs:
                name = label_map.get(str(idx), f"Food_{idx}").replace("_", " ").title()
                top_predictions.append((name, float(preds[idx]) * 100))
            food_name = top_predictions[0][0]
            conf      = top_predictions[0][1]
    else:
        return "Unknown", 0.0, 200.0

    if portion_estimator is not None:
        est_grams = max(50.0, float(portion_estimator.predict(img_tensor, verbose=0)[0][0]))
    else:
        est_grams = max(50.0, float(area_ratio) * 700.0) if area_ratio is not None else 200.0

    return food_name, conf, est_grams


def nms_boxes(boxes, overlap_threshold=0.5):
    """Non-maximum suppression to remove duplicate bounding boxes."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1]), reverse=True)
    keep = []
    for box1 in boxes:
        x1a, y1a, x2a, y2a = box1['box']
        area_a = (x2a - x1a) * (y2a - y1a)
        is_dup = False
        for box2 in keep:
            x1b, y1b, x2b, y2b = box2['box']
            xi = max(x1a, x1b); yi = max(y1a, y1b)
            xa = min(x2a, x2b); ya = min(y2a, y2b)
            if xa > xi and ya > yi:
                ai  = (xa - xi) * (ya - yi)
                ab  = (x2b - x1b) * (y2b - y1b)
                iou = ai / (area_a + ab - ai) if (area_a + ab - ai) > 0 else 0
                if iou > overlap_threshold:
                    is_dup = True
                    break
        if not is_dup:
            keep.append(box1)
    return keep


def detect_food_regions(image, classifier_model, label_map_dict, grid_size=3):
    """Fallback grid-based food detection when YOLO finds nothing."""
    detections = []
    w, h = image.size
    cw, ch = w // grid_size, h // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            x1, y1, x2, y2 = j * cw, i * ch, j * cw + cw, i * ch + ch
            crop = image.crop((x1, y1, x2, y2))
            food_name, conf, _ = classify_crop(crop)
            if food_name != "Unknown" and conf > 0.5:
                detections.append({
                    'food': food_name, 'box': (x1, y1, x2, y2),
                    'conf': conf, 'source': 'grid'
                })
    return detections


def process_image(image, language):
    """Main pipeline: YOLO detection → CNN classification → nutrition lookup → PDF."""
    if image is None:
        return None, "Please upload an image first.", None, None, None

    detections_res = []
    boxes          = []
    yolo_results   = []

    if object_detector is not None:
        results = object_detector(image, conf=max(0.4, YOLO_MIN_CONF))[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id   = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = results.names[class_id] if class_id in results.names else f"Object_{class_id}"
            boxes.append((x1, y1, x2, y2))
            yolo_results.append({'box': (x1, y1, x2, y2), 'class': class_name, 'conf': conf})
        yolo_results = nms_boxes(yolo_results, 0.4)
        boxes        = [r['box'] for r in yolo_results]

    grid_detections = detect_food_regions(image, classifier, label_map, 4)

    if not boxes:
        w, h         = image.size
        boxes        = [(0, 0, w, h)]
        yolo_results = [{'box': (0, 0, w, h), 'class': 'meal', 'conf': 0.95}]

    non_food = {
        'fork', 'knife', 'spoon', 'plate', 'bowl', 'dining table', 'person',
        'potted plant', 'motorcycle', 'bicycle', 'car', 'bus', 'train',
        'dog', 'cat', 'bird', 'mouse', 'chair', 'couch'
    }

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        if x2 <= x1 or y2 <= y1:
            continue
        yc    = yolo_results[idx]['class'] if idx < len(yolo_results) else None
        yconf = yolo_results[idx]['conf']  if idx < len(yolo_results) else 0.0

        if yc and yc.lower() in non_food:
            continue

        crop_img = image.crop((x1, y1, x2, y2))
        try:
            iw, ih = image.size
            ar = (x2 - x1) * (y2 - y1) / float(iw * ih)
        except Exception:
            ar = None

        food_name = "Unknown"
        conf      = 0.0
        est_grams = 200.0

        if yc and yc != 'meal':
            mapped = None
            if yc.lower() in yolo_to_food_mapping:
                mapped = next((o for o in yolo_to_food_mapping[yc.lower()] if o), None)
            food_name = mapped if mapped else yc
            conf      = yconf * 100
            est_grams = max(50.0, float(ar) * 700.0) if ar else 200.0
        else:
            food_name, conf, est_grams = classify_crop(crop_img, ar)

        if FORCE_ACCEPT_DETECTIONS or (food_name != "Unknown" and conf > MIN_CLASS_CONF):
            detections_res.append({
                "food": food_name, "conf": conf,
                "grams": est_grams, "box": (x1, y1, x2, y2)
            })

    for gd in grid_detections:
        if gd['food'] != "Unknown" and gd['conf'] > 0.5:
            gb = gd['box']
            is_dup = any(
                gb[2] > e['box'][0] and gb[0] < e['box'][2] and
                gb[3] > e['box'][1] and gb[1] < e['box'][3]
                for e in detections_res
            )
            if not is_dup:
                ga = (gb[2] - gb[0]) * (gb[3] - gb[1])
                ia = image.size[0] * image.size[1]
                detections_res.append({
                    "food": gd['food'], "conf": gd['conf'],
                    "grams": max(50.0, ga / ia * 700.0), "box": gb
                })

    compute_list    = [{"food": d["food"], "grams": d["grams"]} for d in detections_res]
    result          = engine.compute_plate_nutrition(compute_list)
    total_nutrients = result["total"]

    annotated_img = image.copy()
    draw          = ImageDraw.Draw(annotated_img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    for d in detections_res:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline="#27ae60", width=4)
        text = f'{d["food"]} {d["grams"]:.0f}g'
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([bbox[0], bbox[1], bbox[2] + 10, bbox[3] + 10], fill="#ffffff", outline="#27ae60")
        draw.text((x1 + 5, y1 + 5), text, fill="#27ae60", font=font)

    nutrient_html = build_nutrient_html(total_nutrients, detections_res)

    from PIL import Image as PILImage
    macro_buf = plot_macronutrients(total_nutrients, language)
    cal_buf   = plot_calorie_distribution(total_nutrients, language)
    macro_img = PILImage.open(macro_buf)
    cal_img   = PILImage.open(cal_buf)

    text_report = "Detected Items:\n"
    for d in detections_res:
        text_report += f"- {d['food']} ({d['conf']:.1f}% conf, {d['grams']:.0f} g)\n"
    text_report += engine.format_nutrition_label(total_nutrients)

    if language == "Tamil":
        text_report = translate_to_tamil(text_report)

    out_dir  = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = generate_pdf_report(text_report, None, str(out_dir / "nutrition_report.pdf"))

    return annotated_img, nutrient_html, macro_img, cal_img, pdf_path


# ─── Meal Log Functions ────────────────────────────────────────────────────────
def log_meal_entry(food_name, grams, meal_type):
    if not food_name.strip():
        return get_meal_log_html(), get_daily_summary_html()
    try:
        nutrients_100g = engine.get_nutrition_per_100g(food_name.strip().lower())
        scale   = float(grams) / 100.0
        scaled  = {k: v * scale for k, v in nutrients_100g.items()}
    except Exception:
        scaled = {k: 0 for k in DRI}

    meal_log.append({
        "food":      food_name.strip().title(),
        "grams":     float(grams),
        "meal_type": meal_type,
        "time":      datetime.datetime.now().strftime("%H:%M"),
        "nutrients": scaled,
    })
    return get_meal_log_html(), get_daily_summary_html()


def get_meal_log_html():
    if not meal_log:
        return '<div style="text-align:center;color:var(--text-muted);padding:40px;">No meals logged yet. Add your first meal above!</div>'
    rows   = ""
    colors = {"Breakfast": "var(--accent3)", "Lunch": "var(--accent)", "Snack": "var(--accent2)", "Dinner": "var(--fiber)"}
    for entry in reversed(meal_log):
        col   = colors.get(entry["meal_type"], "var(--accent)")
        cals = entry["nutrients"].get("Calories (kcal)", 0)
        rows += f"""
        <div class="log-entry surface-card" style="margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;">
          <div class="log-left">
            <span class="log-food" style="font-family:'Syne',sans-serif;font-weight:700;">{entry['food']}</span>
            <span class="log-pill" style="border:1px solid {col}; color:{col}; padding:2px 8px; border-radius:12px; font-size:0.75rem; margin-left:10px;">{entry['meal_type']}</span>
          </div>
          <div class="log-right" style="text-align:right;">
            <div class="log-meta" style="color:var(--text-muted);font-size:0.85rem;">{entry['grams']:.0f} g &nbsp;|&nbsp; <strong style="color:var(--text);">{cals:.0f} kcal</strong></div>
            <div class="log-time" style="color:var(--text-muted);font-size:0.75rem;">{entry['time']}</div>
          </div>
        </div>"""
    return f'<div class="meal-log-container">{rows}</div>'


def get_daily_summary_html():
    if not meal_log:
        return '<div style="color:var(--text-muted);text-align:center;">Log meals to see your daily summary.</div>'
    totals = {k: 0.0 for k in DRI}
    for entry in meal_log:
        for k in totals:
            totals[k] += entry["nutrients"].get(k, 0)
    rows = ""
    for key, (label, unit, color, _) in NUTRIENT_META.items():
        val = totals[key]
        dri = DRI[key]
        pct = min(100, round(val / dri * 100))
        rows += f"""
        <div class="dv-row" style="display:flex;align-items:center;margin-bottom:10px;">
            <div class="dv-lbl" style="width:100px;font-size:0.85rem;">{label}</div>
            <div class="dv-val" style="width:70px;text-align:right;margin-right:10px;font-size:0.85rem;">{val:.1f}{unit}</div>
            <div class="dv-bar" style="flex:1;"><div class="progress-track" style="height:6px;background:var(--bg);border-radius:3px;"><div class="progress-fill" style="width:{pct}%;background:{color};height:100%;"></div></div></div>
            <div class="dv-pct" style="width:40px;text-align:right;font-size:0.85rem;">{pct}%</div>
        </div>"""
    return f"""
<div class="daily-summary">
  <div class="syne-header" style="font-size:1.2rem;margin-bottom:15px;color:var(--text);">Today's Totals</div>
  {rows}
</div>"""


def clear_meal_log():
    meal_log.clear()
    return get_meal_log_html(), get_daily_summary_html()


def get_meal_plan_html(goal):
    plan   = MEAL_PLANS.get(goal, MEAL_PLANS["Balanced Health"])
    icons  = {"Breakfast": "🌅", "Lunch": "☀️", "Snack": "🍎", "Dinner": "🌙"}
    html   = f'<div class="plan-container"><h3 class="syne-header" style="font-size:1.5rem;color:var(--accent);margin-bottom:20px;">{goal} Plan</h3>'
    for meal_type, items in plan.items():
        html += f'<div class="surface-card plan-card" style="margin-bottom:15px;">'
        html += f'<div class="plan-title" style="font-family:\'Syne\',sans-serif;font-weight:700;margin-bottom:10px;">{icons.get(meal_type, "")} {meal_type}</div>'
        for item in items:
            html += f'<div class="plan-item" style="color:var(--text-muted);font-size:0.9rem;margin-bottom:5px;">• {item}</div>'
        html += '</div>'
    html += '</div>'
    return html


def get_nutrition_db_html(search_term=""):
    try:
        csv_path = base_dir / "data" / "raw" / "food_nutrition.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            usda_path = base_dir / "data" / "raw" / "usda_nutrients.csv"
            df = pd.read_csv(str(usda_path))
    except Exception:
        return '<div style="color:var(--text-muted);text-align:center;padding:40px;">Database not available.</div>'

    if search_term and search_term.strip():
        mask = df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        df   = df[mask]

    df = df.head(30)
    if df.empty:
        return '<div style="color:var(--text-muted);text-align:center;padding:40px;">No results found.</div>'

    headers   = "".join(f'<th>{c}</th>' for c in df.columns[:8])
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        cells = "".join(f'<td>{str(v)[:30]}</td>' for v in list(row)[:8])
        rows_html += f'<tr>{cells}</tr>'

    label = f'for "{search_term}"' if search_term else "(first 30 entries)"
    return f"""
<div class="db-container fade-in-scroll">
  <div style="color:var(--text-muted);font-size:0.85rem;margin-bottom:10px;">Showing {len(df)} results {label}</div>
  <table class="db-table">
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


def get_diet_insights_html():
    if not meal_log:
        return """
<div style="text-align:center;padding:60px 20px;">
  <div style="font-size:4rem;margin-bottom:20px;opacity:0.2;">📊</div>
  <div class="syne-header" style="font-size:1.3rem;color:var(--text);margin-bottom:10px;">No Data Yet</div>
  <div style="color:var(--text-muted);">Log meals in the Tracker tab to see personalized insights.</div>
</div>"""

    totals = {k: 0.0 for k in DRI}
    for entry in meal_log:
        for k in totals:
            totals[k] += entry["nutrients"].get(k, 0)

    cals    = totals["Calories (kcal)"]
    protein = totals["Protein (g)"]
    carbs   = totals["Carbs (g)"]
    fat     = totals["Fat (g)"]

    score = 0
    if 1600 <= cals <= 2200:           score += 25
    if protein >= 40:                  score += 25
    if totals["Fiber (g)"] >= 20:      score += 25
    if totals["Sodium (mg)"] <= 2000:  score += 25

    score_color = "var(--accent)" if score >= 75 else "var(--accent2)" if score >= 50 else "var(--fat)"
    score_label = "Excellent" if score >= 75 else "Good" if score >= 50 else "Needs Work"

    meal_counts = {}
    for e in meal_log:
        meal_counts[e["meal_type"]] = meal_counts.get(e["meal_type"], 0) + 1

    meal_dist_html = "".join(
        f'<div class="dist-row" style="display:flex;justify-content:space-between;padding:10px;background:var(--bg);border-radius:8px;margin-bottom:8px;"><span>{t}</span><strong style="color:var(--accent);">{c} meals</strong></div>'
        for t, c in meal_counts.items()
    )

    suggestions = []
    if cals < 1500:                     suggestions.append("⚠️ You may be under-eating. Aim for at least 1600 kcal.")
    if cals > 2500:                     suggestions.append("⚠️ Calorie intake is high. Consider reducing portions.")
    if protein < 40:                    suggestions.append("💪 Increase protein intake. Try eggs, chicken, legumes.")
    if totals["Fiber (g)"] < 15:        suggestions.append("🌿 Eat more fiber. Add vegetables, fruits, whole grains.")
    if totals["Sugar (g)"] > 50:        suggestions.append("🍬 Sugar intake is high. Reduce sugary drinks/sweets.")
    if totals["Sodium (mg)"] > 2000:    suggestions.append("🧂 Sodium is elevated. Reduce processed foods.")
    if not suggestions:                 suggestions.append("✅ Great job! Your diet looks balanced.")

    sugg_html = "".join(f'<div class="sugg-row" style="background:var(--bg);border-left:3px solid var(--accent);padding:10px;border-radius:0 8px 8px 0;margin-bottom:8px;font-size:0.85rem;">{s}</div>' for s in suggestions)

    return f"""
<div class="analytics-container fade-in-scroll">
  <div class="kpi-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin-bottom:30px;">
    <div class="surface-card kpi-card" style="border-bottom:3px solid var(--cal);text-align:center;">
      <div class="kpi-val" style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;">{cals:.0f}</div><div class="kpi-lbl" style="color:var(--text-muted);font-size:0.8rem;">kcal</div>
    </div>
    <div class="surface-card kpi-card" style="border-bottom:3px solid var(--protein);text-align:center;">
      <div class="kpi-val" style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;">{protein:.0f}g</div><div class="kpi-lbl" style="color:var(--text-muted);font-size:0.8rem;">Protein</div>
    </div>
    <div class="surface-card kpi-card" style="border-bottom:3px solid var(--carbs);text-align:center;">
      <div class="kpi-val" style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;">{carbs:.0f}g</div><div class="kpi-lbl" style="color:var(--text-muted);font-size:0.8rem;">Carbs</div>
    </div>
    <div class="surface-card kpi-card" style="border-bottom:3px solid var(--fat);text-align:center;">
      <div class="kpi-val" style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;">{fat:.0f}g</div><div class="kpi-lbl" style="color:var(--text-muted);font-size:0.8rem;">Fat</div>
    </div>
    <div class="surface-card kpi-card" style="border-bottom:3px solid {score_color};text-align:center;">
      <div class="kpi-val" style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;color:{score_color};">{score}/100</div><div class="kpi-lbl" style="color:var(--text-muted);font-size:0.8rem;">{score_label}</div>
    </div>
  </div>

  <div class="analytics-split" style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div class="surface-card">
      <h3 class="syne-header" style="margin-bottom:15px;">Meal Distribution</h3>
      {meal_dist_html if meal_dist_html else '<p style="color:var(--text-muted);">No distribution data.</p>'}
    </div>
    <div class="surface-card">
      <h3 class="syne-header" style="margin-bottom:15px;">Insights</h3>
      {sugg_html}
    </div>
  </div>
</div>"""

# ─── HTML Layouts ─────────────────────────────────────────────────────────────
HEADER_HTML = """
<div class="sticky-header">
  <div class="header-logo">🌿 NutriVision AI</div>
  <div class="header-badge"><span class="pulse-dot"></span> YOLOv8n &bull; CNN &bull; USDA DB</div>
</div>
"""

HOME_HTML = """
<div class="hero-landing fade-in-scroll">
    <div class="hero-content">
        <h1 class="syne-header hero-title">See What's In<br><span style="color:var(--accent);">Your Food</span></h1>
        <p class="hero-sub">Precision nutrition for the modern era. Experience the future of food analysis with real-time detection and smart dietary insights.</p>
        <div class="hero-stats">
           <div class="stat-pill">500+ Indian Foods</div>
           <div class="stat-pill">7k+ USDA Entries</div>
           <div class="stat-pill">YOLO Core</div>
        </div>
    </div>
</div>
"""

ABOUT_HTML = """
<div class="about-container fade-in-scroll">
    <div style="text-align:center;margin-bottom:50px;">
        <h2 class="syne-header" style="font-size:2.5rem;">ENGINEERING <span style="color:var(--accent);">WELLNESS</span></h2>
        <p style="color:var(--text-muted);">Technical specifications of NutriVision AI v2.2</p>
    </div>
    <div class="about-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
        <div class="surface-card">
            <h3 class="syne-header">Deep Vision Engine</h3>
            <p style="color:var(--text-muted);line-height:1.6;margin-top:10px;">Detection pipeline uses YOLOv8n for rapid object localization, feeding into a Custom CNN Food Classifier trained on 101 food categories.</p>
        </div>
        <div class="surface-card">
            <h3 class="syne-header">Portion Estimation</h3>
            <p style="color:var(--text-muted);line-height:1.6;margin-top:10px;">A dedicated Portion Regression Head calculates geometric area ratios to derive gram-accurate volume measurements based on perspective mapping.</p>
        </div>
        <div class="surface-card" style="grid-column:1/-1;">
            <h3 class="syne-header">Data Integration</h3>
            <p style="color:var(--text-muted);line-height:1.6;margin-top:10px;">Integrated with the USDA Global Nutrition Database, leveraging Fuzzy Levenshtein matching to resolve non-standard food names to structured nutrient profiles.</p>
        </div>
    </div>
</div>
"""

# ─── CSS ──────────────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg: #f5f5f5;
  --surface: #ffffff;
  --border: #e0e0e0;
  --accent: #27ae60;
  --accent-hover: #2ecc71;
  --accent2: #e67e22;
  --accent3: #2980b9;
  --text: #2c3e50;
  --text-muted: #7f8c8d;
  
  --protein: #2980b9;
  --carbs: #e67e22;
  --fat: #c0392b;
  --fiber: #8e44ad;
  --cal: #27ae60;
}

body, html {
  font-family: 'DM Mono', monospace !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-size: 20px !important; /* Scale up entire application */
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.02'/%3E%3C/svg%3E") !important;
}

.syne-header, h1, h2, h3, h4 {
  font-family: 'Syne', sans-serif !important;
  margin: 0;
  font-weight: 700;
  font-size: 1.5rem;
}

/* Make webpage fit screen */
.gradio-container { 
    max-width: 100% !important; 
    width: 100% !important; 
    padding: 0 2% !important; 
    margin: 0 auto !important;
    background: transparent !important; 
    border: none !important; 
}

/* Base Overrides */
gradio-app { background: transparent !important; border: none !important; }
.surface-card { background: var(--surface) !important; border: 1px solid var(--border) !important; box-shadow: 0 4px 15px rgba(0,0,0,0.04) !important; border-radius: 16px !important; padding: 20px !important; }
input, textarea, select { background: var(--bg) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }
button.primary { background: var(--accent) !important; color: #fff !important; font-weight: 700 !important; border-radius: 12px !important; border: none !important; transition: all 0.2s !important; }
button.primary:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(39, 174, 96, 0.2) !important; }
button.secondary { background: transparent !important; border: 1px solid var(--border) !important; color: var(--text) !important; }

/* Header */
.sticky-header { position: sticky; top: 0; z-index: 1000; display: flex; justify-content: space-between; align-items: center; padding: 15px 30px; background: rgba(255,255,255,0.9); backdrop-filter: blur(10px); border-bottom: 1px solid var(--border); margin-bottom: 20px; border-radius: 0 0 20px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.02); }
.header-logo { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: var(--text); }
.header-badge { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: var(--accent); background: rgba(39, 174, 96, 0.1); padding: 6px 12px; border-radius: 20px; border: 1px solid rgba(39, 174, 96, 0.2); }
.pulse-dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.7); animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(39, 174, 96, 0); } 100% { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0); } }

/* Hero / Upload */
.hero-landing { padding: 60px 20px; text-align: center; }
.hero-title { font-size: 5rem; letter-spacing: -2px; margin-bottom: 20px; line-height: 1.1; color: var(--text); }
.hero-sub { font-size: 1.4rem; color: var(--text-muted); max-width: 800px; margin: 0 auto 40px; }
.hero-stats { display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; }
.stat-pill { background: transparent; border: 1px solid var(--border); padding: 12px 24px; border-radius: 30px; font-size: 1.1rem; color: var(--text-muted); }

/* Progress Bars */
.progress-track { background: var(--bg); border: 1px solid var(--border); height: 8px; border-radius: 4px; overflow: hidden; width: 100%; }
.progress-fill { height: 100%; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }

/* Analysis Grid */
.analysis-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
@media (max-width: 900px) { .analysis-grid { grid-template-columns: 1fr; } }

/* Detected Item */
.detected-item { display: flex; align-items: center; gap: 15px; background: var(--bg); border: 1px solid var(--border); padding: 15px; border-radius: 12px; margin-bottom: 10px; }
.detected-emoji { font-size: 2rem; background: var(--surface); width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
.detected-info { flex: 1; }
.detected-name { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: var(--text); }
.detected-meta { font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }
.detected-conf { width: 120px; }
.conf-lbl { font-size: 0.75rem; color: var(--accent); margin-bottom: 4px; text-align: right; font-weight: bold; }

/* CSS Tabs */
.css-tabs { display: flex; flex-wrap: wrap; }
.css-tabs input[type="radio"] { display: none; }
.css-tabs label { padding: 10px 20px; cursor: pointer; font-family: 'Syne', sans-serif; font-weight: 600; color: var(--text-muted); border-bottom: 2px solid transparent; transition: 0.3s; }
.css-tabs input[type="radio"]:checked + label { color: var(--accent); border-bottom: 2px solid var(--accent); }
.tab-content { width: 100%; padding-top: 20px; display: none; animation: fade 0.4s; }
.css-tabs input#tab-macros:checked ~ #content-macros,
.css-tabs input#tab-vitamins:checked ~ #content-vitamins,
.css-tabs input#tab-insights:checked ~ #content-insights { display: block; }

/* Macros */
.macro-summary { text-align: center; }
.macro-total { margin-bottom: 20px; }
.macro-total-val { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; color: var(--text); }
.macro-total-lbl { color: var(--text-muted); }
.macro-segmented-bar { display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin-bottom: 20px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1); }
.macro-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 10px; }
.m-card { background: var(--bg); padding: 15px 10px; border-radius: 10px; text-align: center; border: 1px solid var(--border); }
.m-lbl { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 5px; font-weight: 600; }
.m-val { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--text); }

/* Vitamins & DV */
.vit-row, .dv-row { display: flex; align-items: center; margin-bottom: 12px; }
.vit-lbl, .dv-lbl { width: 110px; font-size: 0.85rem; color: var(--text); font-weight: bold; }
.vit-bar, .dv-bar { flex: 1; margin: 0 15px; }
.vit-val, .dv-pct { width: 60px; text-align: right; font-size: 0.85rem; color: var(--text-muted); }

/* Insights */
.insight-card { display: flex; gap: 15px; background: var(--bg); border: 1px solid var(--border); border-left: 4px solid var(--text-muted); padding: 15px; border-radius: 0 10px 10px 0; margin-bottom: 10px; }
.insight-icon { font-size: 1.5rem; }
.insight-title { font-weight: 700; color: var(--text); margin-bottom: 5px; }
.insight-desc { font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; }

/* Animations */
.fade-in-scroll { animation: fade 0.6s ease-out forwards; }
@keyframes fade { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

/* Gradio specific overrides for tabs */
.tab-nav { background: transparent !important; border: none !important; border-bottom: 1px solid var(--border) !important; margin-bottom: 30px !important; }
.tab-nav button { font-family: 'Syne', sans-serif !important; border: none !important; border-bottom: 2px solid transparent !important; color: var(--text-muted) !important; border-radius: 0 !important; font-size: 1rem !important; }
.tab-nav button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; background: transparent !important; }

/* Tables */
.db-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.db-table th { background: var(--surface); color: var(--accent); padding: 12px; text-align: left; font-family: 'Syne', sans-serif; border-bottom: 2px solid var(--border); }
.db-table td { padding: 12px; border-bottom: 1px solid var(--border); color: var(--text-muted); }
.db-table tr:hover td { background: rgba(0,0,0,0.02); color: var(--text); }

/* Upload Zone glowing */
.upload-zone { border: 2px dashed var(--border) !important; transition: 0.3s !important; background: var(--surface) !important; }
.upload-zone:hover { border-color: var(--accent) !important; box-shadow: 0 0 20px rgba(39, 174, 96, 0.1) !important; }
"""

# ─── Gradio App ────────────────────────────────────────────────────────────────
with gr.Blocks(title="NutriVision AI", css=custom_css) as demo:
    gr.HTML(HEADER_HTML)
    
    with gr.Tabs(elem_classes="tab-nav"):
        # ── Tab 1: Overview ────────────────────────────────────────────────
        with gr.Tab("🏠 OVERVIEW"):
            gr.HTML(HOME_HTML)

        # ── Tab 2: Food Analyzer ───────────────────────────────────────────
        with gr.Tab("📷 ANALYZE"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="fade-in-scroll" style="padding-top:20px;">
                        <h1 class="syne-header" style="font-size:3rem;margin-bottom:15px;line-height:1.1;">See What's In<br><span style="color:var(--accent);">Your Food</span></h1>
                        <p style="color:var(--text-muted);font-size:1rem;margin-bottom:30px;max-width:90%;">Instantly identify foods, estimate portions, and unlock deep nutritional insights using advanced AI.</p>
                        <div style="display:flex;gap:15px;margin-bottom:30px;">
                            <div class="surface-card" style="padding:10px 15px !important;"><div style="font-size:1.2rem;color:var(--text);font-family:'Syne',sans-serif;font-weight:700;">500+</div><div style="font-size:0.75rem;color:var(--text-muted);">Indian Foods</div></div>
                            <div class="surface-card" style="padding:10px 15px !important;"><div style="font-size:1.2rem;color:var(--text);font-family:'Syne',sans-serif;font-weight:700;">7k+</div><div style="font-size:0.75rem;color:var(--text-muted);">USDA Entries</div></div>
                            <div class="surface-card" style="padding:10px 15px !important;"><div style="font-size:1.2rem;color:var(--text);font-family:'Syne',sans-serif;font-weight:700;">V8</div><div style="font-size:0.75rem;color:var(--text-muted);">YOLO Core</div></div>
                        </div>
                        <div style="display:flex;gap:10px;">
                            <span style="border:1px solid var(--border);padding:4px 12px;border-radius:20px;font-size:0.8rem;color:var(--accent);font-weight:bold;">High Protein</span>
                            <span style="border:1px solid var(--border);padding:4px 12px;border-radius:20px;font-size:0.8rem;color:var(--accent2);font-weight:bold;">Keto</span>
                            <span style="border:1px solid var(--border);padding:4px 12px;border-radius:20px;font-size:0.8rem;color:var(--accent3);font-weight:bold;">Vegan</span>
                        </div>
                    </div>""")
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Meal", elem_classes="upload-zone")
                    lang_toggle = gr.Radio(["English", "Tamil"], label="Interface Language", value="English", visible=False)
                    portion_slider = gr.Slider(50, 500, value=200, label="Portion Size Range (g)", interactive=True)
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze", variant="primary", elem_classes="primary")
                        clear_btn = gr.Button("Clear", elem_classes="secondary")
            
            with gr.Row(elem_classes="analysis-results", visible=True):
                with gr.Column(scale=2):
                    annotated_out = gr.Image(type="pil", label="AI VISION OVERLAY", visible=True)
                    nutrient_display = gr.HTML()
                
                with gr.Column(scale=1):
                    with gr.Column(elem_classes="surface-card"):
                        gr.Markdown("<h3 class='syne-header'>⚙️ Analysis Settings</h3>")
                        detection_mode = gr.Dropdown(["Precise", "Fast"], value="Precise", label="Detection Mode")
                        cuisine_filter = gr.Dropdown(["All", "Indian", "Global"], value="All", label="Cuisine Filter")
                        bbox_toggle = gr.Checkbox(value=True, label="Show Bounding Boxes")
                        conf_thresh = gr.Slider(0, 100, value=50, label="Confidence Threshold")
                    
                    with gr.Column(elem_classes="surface-card"):
                        gr.Markdown("<h3 class='syne-header'>📥 Export</h3>")
                        pdf_output  = gr.File(label="📄 PDF EXPORT")
                        json_btn = gr.Button("Copy JSON", elem_classes="secondary")
                    
                    with gr.Column(elem_classes="surface-card"):
                        gr.Markdown("<h3 class='syne-header'>🧠 Model Pipeline</h3>")
                        gr.HTML("""
                        <div style="font-size:0.85rem;color:var(--text-muted);line-height:1.6;">
                            <div style="margin-bottom:8px;">&bull; <strong style="color:var(--text);">YOLOv8n:</strong> Bounding box localization</div>
                            <div style="margin-bottom:8px;">&bull; <strong style="color:var(--text);">Custom CNN:</strong> 101-class feature extraction</div>
                            <div style="margin-bottom:8px;">&bull; <strong style="color:var(--text);">Portion Estimator:</strong> Geometric regression</div>
                            <div>&bull; <strong style="color:var(--text);">USDA FoodData:</strong> Nutritional matching</div>
                        </div>""")
            
            # Hide the old static charts by placing them in an invisible row, since we generate dynamic CSS charts now
            with gr.Row(visible=False):
                macro_chart = gr.Image()
                cal_chart = gr.Image()

            analyze_btn.click(
                fn=process_image,
                inputs=[image_input, lang_toggle],
                outputs=[annotated_out, nutrient_display, macro_chart, cal_chart, pdf_output],
            )

        # ── Tab 3: Nutrition Database ──────────────────────────────────────
        with gr.Tab("🗄️ DATABASE"):
            with gr.Column(elem_classes="surface-card"):
                gr.Markdown("<h3 class='syne-header' style='color:var(--accent);margin-bottom:20px;'>CORE NUTRITION REPOSITORY</h3>")
                with gr.Row():
                    db_search = gr.Textbox(placeholder="Query food item...", label="SEARCH ENGINE", scale=4)
                    db_btn    = gr.Button("EXECUTE", variant="primary", scale=1, elem_classes="primary")
                db_output = gr.HTML()
            db_btn.click(fn=get_nutrition_db_html, inputs=[db_search], outputs=[db_output])
            demo.load(fn=lambda: get_nutrition_db_html(""), outputs=[db_output])

        # ── Tab 4: Meal Tracker ────────────────────────────────────────────
        with gr.Tab("📅 TRACKER"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="surface-card"):
                    gr.Markdown("<h3 class='syne-header' style='color:var(--accent);margin-bottom:20px;'>LOG ENTRY</h3>")
                    mp_food  = gr.Textbox(label="FOOD VECTOR", placeholder="e.g. burger...")
                    mp_grams = gr.Number(label="MASS (G)", value=100, minimum=1)
                    mp_meal  = gr.Dropdown(["Breakfast", "Lunch", "Snack", "Dinner"], label="PHASE", value="Lunch")
                    mp_add   = gr.Button("⚡ COMMIT RECORD", variant="primary", elem_classes="primary")
                    mp_clear = gr.Button("🗑️ PURGE LOG", elem_classes="secondary")
                with gr.Column(scale=2):
                    with gr.Column(elem_classes="surface-card"):
                        mp_log_html = gr.HTML(label="HISTORICAL VECTOR")
                    with gr.Column(elem_classes="surface-card"):
                        mp_summary_html = gr.HTML(label="CUMULATIVE METRICS")

            mp_add.click(fn=log_meal_entry,  inputs=[mp_food, mp_grams, mp_meal], outputs=[mp_log_html, mp_summary_html])
            mp_clear.click(fn=clear_meal_log, outputs=[mp_log_html, mp_summary_html])
            demo.load(fn=lambda: (get_meal_log_html(), get_daily_summary_html()), outputs=[mp_log_html, mp_summary_html])

        # ── Tab 5: Diet Analytics ──────────────────────────────────────────
        with gr.Tab("📈 ANALYTICS"):
            with gr.Column():
                refresh_btn   = gr.Button("🔄 RECALCULATE INSIGHTS", variant="primary", elem_classes="primary")
                insights_html = gr.HTML()
            refresh_btn.click(fn=get_diet_insights_html, outputs=[insights_html])
            demo.load(fn=get_diet_insights_html, outputs=[insights_html])

        # ── Tab 6: Meal Blueprints ─────────────────────────────────────────
        with gr.Tab("🍽️ BLUEPRINTS"):
            with gr.Column(elem_classes="surface-card"):
                gr.Markdown("<h3 class='syne-header' style='margin-bottom:20px;'>OPTIMIZATION BLUEPRINTS</h3>")
                plan_goal = gr.Radio(["Weight Loss", "Muscle Gain", "Balanced Health"], label="TARGET GOAL", value="Balanced Health")
                plan_out  = gr.HTML()
            plan_goal.change(fn=get_meal_plan_html, inputs=[plan_goal], outputs=[plan_out])
            demo.load(fn=lambda: get_meal_plan_html("Balanced Health"), outputs=[plan_out])

        # ── Tab 7: Architecture ────────────────────────────────────────────
        with gr.Tab("ℹ️ ARCHITECTURE"):
            gr.HTML(ABOUT_HTML)

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    demo.launch(server_port=port, share=False)