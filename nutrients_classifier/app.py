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
classifier_path = base_dir / "models" / "food_classifier_custom.h5"
portion_path    = base_dir / "models" / "portion_estimator.h5"
label_map_path  = base_dir / "data" / "processed" / "label_map.json"
yolo_mapping_path = base_dir / "data" / "processed" / "yolo_to_nutrition_mapping.json"

engine = NutritionEngine(base_dir / "data")

yolo_to_food_mapping = {}
try:
    with open(yolo_mapping_path) as f:
        yolo_to_food_mapping = json.load(f)
except Exception as e:
    print(f"Could not load YOLO mapping: {e}")

DEBUG_DETAILED = os.environ.get("DEBUG_DETAILED", "0") == "1"
FORCE_ACCEPT_DETECTIONS = os.environ.get("FORCE_ACCEPT_DETECTIONS", "0") == "1"
MIN_CLASS_CONF = float(os.environ.get("MIN_CLASS_CONF", "0.01"))
YOLO_MIN_CONF = float(os.environ.get("YOLO_MIN_CONF", "0.05"))

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
        label_map = json.load(f)
except Exception:
    label_map = {str(i): f"Food_{i}" for i in range(101)}

try:
    from ultralytics import YOLO
    object_detector = YOLO("yolov8n.pt")
    print("YOLOv8 model loaded.")
except Exception as e:
    print(f"Could not load YOLO: {e}.")
    object_detector = None

fallback_model = None
_mobilenet_preprocess = None
_mobilenet_decode = None
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
    "Calories (kcal)": ("🔥 Calories",      "kcal", "#e74c3c", (300, 700)),
    "Protein (g)":     ("💪 Protein",       "g",    "#2980b9", (10,  25)),
    "Fat (g)":         ("🥑 Fat",           "g",    "#f39c12", (10,  25)),
    "Carbs (g)":       ("🍞 Carbohydrates", "g",    "#8e44ad", (30,  80)),
    "Fiber (g)":       ("🌿 Fiber",         "g",    "#27ae60", (5,   15)),
    "Sugar (g)":       ("🍬 Sugar",         "g",    "#e67e22", (5,   20)),
    "Sodium (mg)":     ("🧂 Sodium",        "mg",   "#95a5a6", (300, 800)),
    "Vitamin C (mg)":  ("🍊 Vitamin C",     "mg",   "#f1c40f", (15,  60)),
    "Calcium (mg)":    ("🦴 Calcium",       "mg",   "#1abc9c", (200, 600)),
    "Iron (mg)":       ("⚙️ Iron",          "mg",   "#7f8c8d", (3,   10)),
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
        f'<div style="background:#e8e8e8;border-radius:6px;height:10px;width:100%;">'
        f'<div style="width:{pct}%;background:{color};border-radius:6px;height:10px;"></div>'
        f'</div>'
    )

def _nutrient_level_label(value, thresholds):
    if value <= thresholds[0]:
        return "Low", "#27ae60"
    elif value <= thresholds[1]:
        return "Medium", "#f39c12"
    else:
        return "High", "#e74c3c"

def build_nutrient_html(nutrients, detections_list):
    preds_html = ""
    for d in detections_list:
        preds_html += f"""
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);border-radius:12px;padding:12px 16px;margin-bottom:10px;color:white;display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-size:1.2rem;font-weight:800;">{d['food']}</div>
            <div style="font-size:0.9rem;opacity:0.9;">⚖️ Portion: <strong>{d['grams']:.0f} g</strong> | 🎯 Conf: <strong>{d['conf']:.1f}%</strong></div>
          </div>
        </div>"""

    nutrient_rows = ""
    for key, (label, unit, color, thresholds) in NUTRIENT_META.items():
        val = nutrients.get(key, 0)
        dri = DRI.get(key, 1)
        pct_dri = min(100, round((val / dri) * 100))
        level_label, level_color = _nutrient_level_label(val, thresholds)
        bar = _level_bar(val, dri, color)
        nutrient_rows += f"""
        <tr>
          <td style="padding:8px 6px;font-weight:600;color:#2c3e50;white-space:nowrap;">{label}</td>
          <td style="padding:8px 6px;text-align:right;font-weight:700;color:#333;">{val:.1f} {unit}</td>
          <td style="padding:8px 12px;width:140px;">{bar}</td>
          <td style="padding:8px 6px;text-align:center;">
            <span style="background:{level_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:700;">{level_label}</span>
          </td>
          <td style="padding:8px 6px;color:#888;font-size:0.8rem;text-align:right;">{pct_dri}% DV</td>
        </tr>"""

    # Tips & Alternatives
    tips_html = ""
    shown_foods = set()
    for d in detections_list:
        food_key = d["food"].lower().replace(" ", "_")
        if food_key in FOOD_TIPS and food_key not in shown_foods:
            shown_foods.add(food_key)
            tips_html += f'<div style="background:#fff3cd;border-left:4px solid #f39c12;padding:10px 14px;border-radius:8px;margin-bottom:8px;"><strong>💡 Tip:</strong> {FOOD_TIPS[food_key]}</div>'
            if food_key in HEALTHY_ALTERNATIVES:
                alts = ", ".join(HEALTHY_ALTERNATIVES[food_key])
                tips_html += f'<div style="background:#d4edda;border-left:4px solid #27ae60;padding:10px 14px;border-radius:8px;margin-bottom:8px;"><strong>🥦 Healthier Alternatives:</strong> {alts}</div>'

    tips_section = f"""
    <div style="background:#fff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 20px;margin-top:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
      <div style="font-weight:700;font-size:1rem;color:#2c3e50;margin-bottom:12px;">💚 Health Tips & Alternatives</div>
      {tips_html if tips_html else '<p style="color:#888;">No specific tips for detected items.</p>'}
    </div>""" if tips_html else ""

    return f"""
<div style="font-family:'Inter',sans-serif;max-width:760px;">
  <div style="background:#fff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 20px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <div style="font-weight:700;font-size:1.1rem;color:#2c3e50;margin-bottom:14px;">🍽️ Detected Items</div>
    {preds_html}
  </div>
  <div style="background:#fff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <div style="font-weight:700;font-size:1rem;color:#2c3e50;margin-bottom:12px;">🧬 Cumulative Nutrient Levels</div>
    <table style="width:100%;border-collapse:collapse;">
      <thead><tr style="border-bottom:2px solid #eee;">
        <th style="padding:6px;text-align:left;color:#888;font-size:0.8rem;">NUTRIENT</th>
        <th style="padding:6px;text-align:right;color:#888;font-size:0.8rem;">AMOUNT</th>
        <th style="padding:6px;color:#888;font-size:0.8rem;">LEVEL</th>
        <th style="padding:6px;text-align:center;color:#888;font-size:0.8rem;">STATUS</th>
        <th style="padding:6px;text-align:right;color:#888;font-size:0.8rem;">DAILY %</th>
      </tr></thead>
      <tbody>{nutrient_rows}</tbody>
    </table>
  </div>
  {tips_section}
</div>"""

def classify_crop(crop_img, area_ratio=None):
    img_resized = crop_img.resize((IMG_SIZE, IMG_SIZE))
    img_array  = np.array(img_resized) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    top_predictions = []
    if classifier is not None:
        preds = classifier.predict(img_tensor, verbose=0)[0]
        max_prob = float(preds.max()) if hasattr(preds, 'max') else 0.0
        if max_prob < 0.02:
            food_name = "Unknown"; conf = max_prob * 100
        else:
            top_idxs = preds.argsort()[::-1][:3]
            for idx in top_idxs:
                name = label_map.get(str(idx), f"Food_{idx}").replace("_", " ").title()
                top_predictions.append((name, float(preds[idx]) * 100))
            food_name = top_predictions[0][0]; conf = top_predictions[0][1]
    else:
        return "Unknown", 0.0, []
    if portion_estimator is not None:
        est_grams = max(50.0, float(portion_estimator.predict(img_tensor, verbose=0)[0][0]))
    else:
        est_grams = max(50.0, float(area_ratio) * 700.0) if area_ratio is not None else 200.0
    return food_name, conf, est_grams

def nms_boxes(boxes, overlap_threshold=0.5):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: (b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1]), reverse=True)
    keep = []
    for box1 in boxes:
        x1a,y1a,x2a,y2a = box1['box']; area_a=(x2a-x1a)*(y2a-y1a)
        is_dup = False
        for box2 in keep:
            x1b,y1b,x2b,y2b = box2['box']
            xi = max(x1a,x1b); yi = max(y1a,y1b); xa = min(x2a,x2b); ya = min(y2a,y2b)
            if xa>xi and ya>yi:
                ai=(xa-xi)*(ya-yi); ab=(x2b-x1b)*(y2b-y1b)
                iou=ai/(area_a+ab-ai) if (area_a+ab-ai)>0 else 0
                if iou>overlap_threshold: is_dup=True; break
        if not is_dup: keep.append(box1)
    return keep

def detect_food_regions(image, classifier, label_map, grid_size=3):
    detections = []; w,h = image.size
    cw,ch = w//grid_size, h//grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            x1,y1,x2,y2 = j*cw, i*ch, j*cw+cw, i*ch+ch
            crop = image.crop((x1,y1,x2,y2))
            food_name,conf,_ = classify_crop(crop)
            if food_name!="Unknown" and conf>0.5:
                detections.append({'food':food_name,'box':(x1,y1,x2,y2),'conf':conf,'source':'grid'})
    return detections

def process_image(image, language):
    if image is None:
        return None, "Please upload an image first.", None, None, None
    detections_res = []; boxes = []; yolo_results = []
    if object_detector is not None:
        results = object_detector(image, conf=max(0.4, YOLO_MIN_CONF))[0]
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0]); conf = float(box.conf[0])
            class_name = results.names[class_id] if class_id in results.names else f"Object_{class_id}"
            boxes.append((x1,y1,x2,y2))
            yolo_results.append({'box':(x1,y1,x2,y2),'class':class_name,'conf':conf})
        yolo_results = nms_boxes(yolo_results, 0.4)
        boxes = [r['box'] for r in yolo_results]
    grid_detections = detect_food_regions(image, classifier, label_map, 4)
    if not boxes:
        w,h = image.size; boxes=[(0,0,w,h)]
        yolo_results=[{'box':(0,0,w,h),'class':'meal','conf':0.95}]
    non_food = ['fork','knife','spoon','plate','bowl','dining table','person','potted plant',
                'motorcycle','bicycle','car','bus','train','dog','cat','bird','mouse','chair','couch']
    for idx,(x1,y1,x2,y2) in enumerate(boxes):
        if x2<=x1 or y2<=y1: continue
        yc = yolo_results[idx]['class'] if idx<len(yolo_results) else None
        yconf = yolo_results[idx]['conf'] if idx<len(yolo_results) else 0.0
        if yc and yc.lower() in non_food: continue
        crop_img = image.crop((x1,y1,x2,y2))
        try:
            iw,ih=image.size; ar=(x2-x1)*(y2-y1)/float(iw*ih)
        except: ar=None
        food_name="Unknown"; conf=0.0; est_grams=200.0
        if yc and yc!='meal':
            mapped = None
            if yc.lower() in yolo_to_food_mapping:
                mapped = next((o for o in yolo_to_food_mapping[yc.lower()] if o), None)
            food_name = mapped if mapped else yc
            conf = yconf; est_grams = max(50.0,float(ar)*700.0) if ar else 200.0
        else:
            food_name,conf,est_grams = classify_crop(crop_img, ar)
        if FORCE_ACCEPT_DETECTIONS or (food_name!="Unknown" and conf>MIN_CLASS_CONF):
            detections_res.append({"food":food_name,"conf":conf,"grams":est_grams,"box":(x1,y1,x2,y2)})
    for gd in grid_detections:
        if gd['food']!="Unknown" and gd['conf']>0.5:
            gb=gd['box']
            is_dup = any(gb[2]>e['box'][0] and gb[0]<e['box'][2] and gb[3]>e['box'][1] and gb[1]<e['box'][3] for e in detections_res)
            if not is_dup:
                ga=(gb[2]-gb[0])*(gb[3]-gb[1]); ia=image.size[0]*image.size[1]
                detections_res.append({"food":gd['food'],"conf":gd['conf'],"grams":max(50.0,ga/ia*700.0),"box":gb})
    compute_list = [{"food":d["food"],"grams":d["grams"]} for d in detections_res]
    result = engine.compute_plate_nutrition(compute_list)
    total_nutrients = result["total"]
    annotated_img = image.copy(); draw = ImageDraw.Draw(annotated_img)
    try: font = ImageFont.truetype("arial.ttf", 24)
    except: font = ImageFont.load_default()
    for d in detections_res:
        x1,y1,x2,y2 = d["box"]; draw.rectangle([x1,y1,x2,y2],outline="#667eea",width=5)
        text=f'{d["food"]} {d["grams"]:.0f}g'; bbox=draw.textbbox((x1,y1),text,font=font)
        draw.rectangle([bbox[0],bbox[1],bbox[2]+10,bbox[3]+10],fill="#667eea")
        draw.text((x1+5,y1+5),text,fill="white",font=font)
    nutrient_html = build_nutrient_html(total_nutrients, detections_res)
    from PIL import Image as PILImage
    macro_buf = plot_macronutrients(total_nutrients, language)
    cal_buf   = plot_calorie_distribution(total_nutrients, language)
    macro_img = PILImage.open(macro_buf); cal_img = PILImage.open(cal_buf)
    text_report = "Detected Items:\n"
    for d in detections_res:
        text_report += f"- {d['food']} ({d['conf']:.1f}% conf, {d['grams']:.0f} g)\n"
    text_report += engine.format_nutrition_label(total_nutrients)
    if language == "Tamil":
        text_report = translate_to_tamil(text_report)
    out_dir = base_dir/"outputs"; out_dir.mkdir(parents=True,exist_ok=True)
    pdf_path = generate_pdf_report(text_report, None, str(out_dir/"nutrition_report.pdf"))
    return annotated_img, nutrient_html, macro_img, cal_img, pdf_path

def log_meal_entry(food_name, grams, meal_type):
    if not food_name.strip():
        return get_meal_log_html(), get_daily_summary_html()
    try:
        nutrients_100g = engine.get_nutrition_per_100g(food_name.strip().lower())
        scale = float(grams) / 100.0
        scaled = {k: v*scale for k,v in nutrients_100g.items()}
    except Exception:
        scaled = {k: 0 for k in DRI}
    meal_log.append({
        "food": food_name.strip().title(),
        "grams": float(grams), "meal_type": meal_type,
        "time": datetime.datetime.now().strftime("%H:%M"),
        "nutrients": scaled
    })
    return get_meal_log_html(), get_daily_summary_html()

def get_meal_log_html():
    if not meal_log:
        return '<div style="text-align:center;color:#888;padding:40px;">No meals logged yet. Add your first meal above! 🍽️</div>'
    rows = ""
    colors = {"Breakfast":"#e8f5e9","Lunch":"#e3f2fd","Snack":"#fff8e1","Dinner":"#fce4ec"}
    for entry in reversed(meal_log):
        bg = colors.get(entry["meal_type"], "#f5f5f5")
        cals = entry["nutrients"].get("Calories (kcal)",0)
        rows += f"""
        <div style="background:{bg};border-radius:12px;padding:12px 16px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;">
          <div>
            <span style="font-weight:700;font-size:1rem;color:#2c3e50;">{entry['food']}</span>
            <span style="margin-left:10px;background:#667eea;color:white;padding:2px 10px;border-radius:20px;font-size:0.75rem;">{entry['meal_type']}</span>
          </div>
          <div style="text-align:right;color:#555;font-size:0.9rem;">
            <div>{entry['grams']:.0f} g &nbsp;|&nbsp; <strong>{cals:.0f} kcal</strong></div>
            <div style="color:#aaa;font-size:0.8rem;">{entry['time']}</div>
          </div>
        </div>"""
    return f'<div style="font-family:Inter,sans-serif;max-height:400px;overflow-y:auto;">{rows}</div>'

def get_daily_summary_html():
    if not meal_log:
        return '<div style="color:#888;text-align:center;">Log meals to see your daily summary.</div>'
    totals = {k: 0.0 for k in DRI}
    for entry in meal_log:
        for k in totals:
            totals[k] += entry["nutrients"].get(k, 0)
    rows = ""
    for key, (label,unit,color,_) in NUTRIENT_META.items():
        val = totals[key]; dri = DRI[key]
        pct = min(100, round(val/dri*100))
        bar = _level_bar(val, dri, color)
        rows += f"""
        <tr>
          <td style="padding:6px 8px;font-weight:600;color:#2c3e50;white-space:nowrap;">{label}</td>
          <td style="padding:6px 8px;font-weight:700;text-align:right;">{val:.1f} {unit}</td>
          <td style="padding:6px 12px;width:130px;">{bar}</td>
          <td style="padding:6px 8px;color:#888;font-size:0.8rem;">{pct}% DV</td>
        </tr>"""
    return f"""
<div style="font-family:Inter,sans-serif;">
  <div style="font-weight:700;font-size:1rem;color:#2c3e50;margin-bottom:10px;">📊 Today's Totals</div>
  <table style="width:100%;border-collapse:collapse;">
    <tbody>{rows}</tbody>
  </table>
</div>"""

def clear_meal_log():
    meal_log.clear()
    return get_meal_log_html(), get_daily_summary_html()

def get_meal_plan_html(goal):
    plan = MEAL_PLANS.get(goal, MEAL_PLANS["Balanced Health"])
    icons = {"Breakfast":"🌅","Lunch":"☀️","Snack":"🍎","Dinner":"🌙"}
    colors = {"Breakfast":"#e8f5e9","Lunch":"#e3f2fd","Snack":"#fff8e1","Dinner":"#fce4ec"}
    html = f'<div style="font-family:Inter,sans-serif;"><div style="font-weight:800;font-size:1.3rem;color:#2c3e50;margin-bottom:16px;">🎯 {goal} Meal Plan</div>'
    for meal_type, items in plan.items():
        bg = colors.get(meal_type,"#f5f5f5")
        html += f'<div style="background:{bg};border-radius:14px;padding:16px 20px;margin-bottom:14px;">'
        html += f'<div style="font-weight:700;font-size:1rem;color:#2c3e50;margin-bottom:10px;">{icons.get(meal_type,"")} {meal_type}</div>'
        for item in items:
            html += f'<div style="background:white;border-radius:8px;padding:8px 14px;margin-bottom:6px;color:#555;">• {item}</div>'
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
            df = pd.read_csv(usda_path)
    except Exception:
        return '<div style="color:#888;text-align:center;padding:40px;">Nutrition database not available.</div>'

    if search_term and search_term.strip():
        mask = df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        df = df[mask]

    df = df.head(30)
    if df.empty:
        return '<div style="color:#888;text-align:center;padding:40px;">No results found.</div>'

    headers = "".join(f'<th style="padding:8px 12px;background:#667eea;color:white;font-weight:600;white-space:nowrap;">{c}</th>' for c in df.columns[:8])
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#f8f9ff" if i % 2 == 0 else "white"
        cells = "".join(f'<td style="padding:8px 12px;color:#444;border-bottom:1px solid #eee;">{str(v)[:30]}</td>' for v in list(row)[:8])
        rows_html += f'<tr style="background:{bg};">{cells}</tr>'

    return f"""
<div style="font-family:Inter,sans-serif;overflow-x:auto;">
  <div style="color:#888;font-size:0.85rem;margin-bottom:10px;">Showing {len(df)} results {f'for "{search_term}"' if search_term else '(first 30 entries)'}</div>
  <table style="width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""

def get_diet_insights_html():
    if not meal_log:
        return """
<div style="font-family:Inter,sans-serif;text-align:center;padding:60px 20px;">
  <div style="font-size:4rem;margin-bottom:20px;">📊</div>
  <div style="font-size:1.3rem;font-weight:700;color:#2c3e50;margin-bottom:10px;">No Data Yet</div>
  <div style="color:#888;">Log meals in the Meal Planner tab to see personalized insights!</div>
</div>"""

    totals = {k: 0.0 for k in DRI}
    for entry in meal_log:
        for k in totals: totals[k] += entry["nutrients"].get(k, 0)

    cals = totals["Calories (kcal)"]; protein = totals["Protein (g)"]
    carbs = totals["Carbs (g)"]; fat = totals["Fat (g)"]

    # Score
    score = 0
    if 1600 <= cals <= 2200: score += 25
    if protein >= 40: score += 25
    if totals["Fiber (g)"] >= 20: score += 25
    if totals["Sodium (mg)"] <= 2000: score += 25

    score_color = "#27ae60" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"
    score_label = "Excellent 🌟" if score >= 75 else "Good 👍" if score >= 50 else "Needs Work 💪"

    meal_counts = {}
    for e in meal_log: meal_counts[e["meal_type"]] = meal_counts.get(e["meal_type"],0)+1
    meal_dist_html = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:6px 10px;background:#f8f9ff;border-radius:8px;margin-bottom:4px;"><span>{t}</span><strong>{c} meal(s)</strong></div>'
        for t,c in meal_dist.items()) if (meal_dist := meal_counts) else ""

    suggestions = []
    if cals < 1500: suggestions.append("⚠️ You may be under-eating. Aim for at least 1600 kcal/day.")
    if cals > 2500: suggestions.append("⚠️ Calorie intake is high. Consider reducing portion sizes.")
    if protein < 40: suggestions.append("💪 Increase protein intake. Try eggs, chicken, legumes.")
    if totals["Fiber (g)"] < 15: suggestions.append("🌿 Eat more fiber. Add vegetables, fruits, and whole grains.")
    if totals["Sugar (g)"] > 50: suggestions.append("🍬 Sugar intake is high. Reduce sugary drinks and sweets.")
    if totals["Sodium (mg)"] > 2000: suggestions.append("🧂 Sodium is elevated. Reduce processed foods and added salt.")
    if not suggestions: suggestions.append("✅ Great job! Your diet looks balanced today.")

    sugg_html = "".join(f'<div style="background:#f0f4ff;border-left:4px solid #667eea;padding:10px 14px;border-radius:8px;margin-bottom:8px;color:#2c3e50;">{s}</div>' for s in suggestions)

    return f"""
<div style="font-family:Inter,sans-serif;">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px;">
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);border-radius:16px;padding:20px;color:white;text-align:center;">
      <div style="font-size:2.5rem;font-weight:800;">{cals:.0f}</div>
      <div style="opacity:0.85;">kcal today</div>
    </div>
    <div style="background:linear-gradient(135deg,#2980b9,#6dd5fa);border-radius:16px;padding:20px;color:white;text-align:center;">
      <div style="font-size:2.5rem;font-weight:800;">{protein:.0f}g</div>
      <div style="opacity:0.85;">Protein</div>
    </div>
    <div style="background:linear-gradient(135deg,#8e44ad,#c0392b);border-radius:16px;padding:20px;color:white;text-align:center;">
      <div style="font-size:2.5rem;font-weight:800;">{carbs:.0f}g</div>
      <div style="opacity:0.85;">Carbs</div>
    </div>
    <div style="background:linear-gradient(135deg,#f39c12,#e74c3c);border-radius:16px;padding:20px;color:white;text-align:center;">
    <div style="background:#222;border-radius:16px;padding:20px;color:white;text-align:center;border:1px solid #444;">
      <div style="font-size:2.5rem;font-weight:800;">{fat:.0f}g</div>
      <div style="opacity:0.85;">Fat</div>
    </div>
    <div style="background:#222;border-radius:16px;padding:20px;color:white;text-align:center;border:1px solid #444;">
      <div style="font-size:2.5rem;font-weight:800;">{score}/100</div>
      <div style="opacity:0.85;">Diet Score — {score_label}</div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div style="background:#fff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 20px;">
      <div style="font-weight:700;color:#2c3e50;margin-bottom:12px;">🍽️ Meal Distribution</div>
      {meal_dist_html}
    </div>
    <div style="background:#fff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 20px;">
      <div style="font-weight:700;color:#2c3e50;margin-bottom:12px;">💡 Personalized Suggestions</div>
      {sugg_html}
    </div>
  </div>
</div>"""

# ─── CSS ──────────────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');

:root {
    --primary: #ffffff;
    --primary-glow: transparent;
    --secondary: #333333;
    --accent: #ffffff;
    --bg-dark: #000000;
    --card-bg: #0a0a0a;
    --card-border: #333333;
    --text-main: #ffffff;
    --text-dim: #a3a3a3;
    color-scheme: dark;
    --body-text-color: #ffffff !important;
    --body-text-color-subdued: #a3a3a3 !important;
    --block-label-text-color: #ffffff !important;
    --block-title-text-color: #ffffff !important;
    --checkbox-label-text-color: #ffffff !important;
    --input-text-color: #ffffff !important;
    --form-text-color: #ffffff !important;
}

gradio-app, .gradio-container, .gradio-container-wrapper {
    background: transparent !important;
}

body {
    background: #000000 !important;
    background-attachment: fixed !important;
    font-family: 'Outfit', sans-serif;
    min-height: 100vh;
    color: var(--text-main);
    margin: 0;
    overflow-x: hidden;
}

.gradio-container::before, .gradio-container::after {
    display: none !important;
}

* {
    box-shadow: none !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 20px !important;
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    background: #111111 !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 20px !important;
    padding: 6px !important;
    margin-bottom: 30px !important;
    display: flex !important;
    justify-content: center !important;
    gap: 8px !important;
    position: sticky !important;
    top: 20px !important;
    z-index: 1000 !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 20px !important;
    border-radius: 14px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border-bottom: 2px solid transparent !important;
}

.tab-nav button.selected {
    background: #ffffff !important;
    color: #000000 !important;
    border-bottom-color: #ffffff !important;
}

button, input, select, textarea {
    outline: none !important;
    --ring-color: transparent !important;
    --primary-500: #ffffff !important;
}

.tab-nav button:hover:not(.selected) {
    background: #333333 !important;
    color: white !important;
}

.glass-card {
    background: #0a0a0a !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 24px !important;
    padding: 24px !important;
    transition: transform 0.3s ease !important;
}

.glass-card:hover {
    transform: translateY(-4px) !important;
    border-color: #666666 !important;
}

h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -3px !important;
    color: #ffffff !important;
    margin-bottom: 8px !important;
    text-align: center !important;
}

.subtitle {
    font-size: 1.1rem !important;
    color: #a3a3a3 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    text-align: center !important;
}

button.primary {
    background: #ffffff !important;
    border: none !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border-radius: 16px !important;
    padding: 16px 32px !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: scale(1.02) translateY(-2px) !important;
    background: #cccccc !important;
}

.gradio-container label span, .gradio-container label {
    color: #ffffff !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    letter-spacing: 1.5px !important;
    font-weight: 700 !important;
    opacity: 0.7 !important;
}

input, textarea, select {
    background: #111111 !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Tab Content Spacing */
.tabitem {
    padding: 20px 0 !important;
    animation: fadeIn 0.5s ease-out !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
"""

HOME_HTML = """
<div style="font-family:'Outfit',sans-serif; color:white; overflow:hidden;">
    <div style="padding: 100px 20px; text-align: center;">
        <div style="font-size: 6rem; margin-bottom: 24px; filter: grayscale(100%);">🥗</div>
        <h2 style="font-size: 4rem; font-weight: 900; letter-spacing: -4px; margin: 0; background: linear-gradient(to right, #fff, #a3a3a3); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">NutriVision AI</h2>
        <p style="font-size: 1.25rem; color: rgba(255,255,255,0.7); max-width: 700px; margin: 20px auto 40px; line-height: 1.6;">Precision nutrition for the modern era. Experience the future of food analysis with real-time detection and smart dietary insights.</p>
        
        <div style="display: flex; justify-content: center; gap: 20px;">
           <div style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 12px 24px; border-radius: 100px; font-weight: 600; font-size: 0.9rem;">⚡ REAL-TIME VIEW</div>
           <div style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 12px 24px; border-radius: 100px; font-weight: 600; font-size: 0.9rem;">🧬 DEEP INSIGHTS</div>
           <div style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 12px 24px; border-radius: 100px; font-weight: 600; font-size: 0.9rem;">🛡️ ENCRYPTED DATA</div>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; padding: 40px 0;">
        <div class="glass-card" style="text-align: left;">
            <div style="font-size: 2.5rem; margin-bottom: 16px; filter: grayscale(100%);">📸</div>
            <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 12px; color: #ffffff;">Vision Analysis</h3>
            <p style="color: #a3a3a3; font-size: 0.95rem; line-height: 1.6;">Snap and analyze. Our YOLO-driven engine identifies multiple foods instantly with high precision.</p>
        </div>
        <div class="glass-card" style="text-align: left;">
            <div style="font-size: 2.5rem; margin-bottom: 16px; filter: grayscale(100%);">📊</div>
            <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 12px; color: #ffffff;">Smart Inventory</h3>
            <p style="color: #a3a3a3; font-size: 0.95rem; line-height: 1.6;">Access over 10,000+ entries from the global nutrition database with intuitive search and filters.</p>
        </div>
        <div class="glass-card" style="text-align: left;">
            <div style="font-size: 2.5rem; margin-bottom: 16px; filter: grayscale(100%);">📈</div>
            <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 12px; color: #ffffff;">Health Score</h3>
            <p style="color: #a3a3a3; font-size: 0.95rem; line-height: 1.6;">Receive daily health scores and personalized recommendations based on your actual intake patterns.</p>
        </div>
    </div>

    <div style="margin-top: 60px; padding: 40px; border-radius: 30px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); text-align: center;">
        <h3 style="font-size: 1.5rem; font-weight: 700; margin-bottom: 8px;">SCALE YOUR NUTRITION</h3>
        <p style="color: rgba(255,255,255,0.6); margin-bottom: 0;">Switch to the <b>ANALYZE</b> tab and upload your meal photo.</p>
    </div>
</div>
"""

ABOUT_HTML = """
<div style="font-family:'Outfit',sans-serif; color:white; max-width:900px; margin: 0 auto; padding: 40px 0;">
    <div style="text-align: center; margin-bottom: 60px;">
        <h2 style="font-size: 3rem; font-weight: 900; letter-spacing: -2px; margin-bottom: 12px;">ENGINEERING <span style="color:#ffffff;">WELLNESS</span></h2>
        <p style="color: #a3a3a3; font-size: 1.1rem;">Technical specifications and system architecture of NutriVision AI v2.2</p>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
        <div class="glass-card">
            <h3 style="color:#ffffff; font-size: 1.2rem; margin-bottom: 16px; font-weight: 700;">Deep Vision Engine</h3>
            <p style="color: #a3a3a3; font-size: 0.95rem; line-height: 1.7;">Our detection pipeline utilizes <b>YOLOv8 Small</b> for rapid object localization, feeding into a <b>MobileNetV2-based Food Classifier</b> fine-tuned on custom datasets.</p>
        </div>
        <div class="glass-card">
            <h3 style="color:#ffffff; font-size: 1.2rem; margin-bottom: 16px; font-weight: 700;">Nuanced Estimation</h3>
            <p style="color: #a3a3a3; font-size: 0.95rem; line-height: 1.7;">A dedicated <b>Portion Estimation Head</b> calculates geometric area ratios to derive gram-accurate volume measurements based on perspective mapping.</p>
        </div>
        <div class="glass-card" style="grid-column: span 2;">
            <h3 style="color:#ffffff; font-size: 1.2rem; margin-bottom: 16px; font-weight: 700;">Data Integration</h3>
            <p style="color: #a3a3a3; font-size: 0.95rem; line-height: 1.7;">Integrated with the <b>USDA Global Nutrition Database</b>, leveraging <b>Fuzzy Levenshtein matching</b> to resolve non-standard food names to structured nutrient profiles.</p>
        </div>
    </div>

    <div style="margin-top: 40px; padding: 24px; border-radius: 20px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); font-size: 0.85rem; color: rgba(255,255,255,0.6); line-height: 1.6; text-align: center;">
        <b>⚠️ MEDICAL DISCLAIMER:</b> NutriVision AI estimates are derived from computational models. Consult a clinical nutritionist for medical dietary prescriptions.
    </div>
</div>
"""

# ─── Gradio App ───────────────────────────────────────────────────────────────
with gr.Blocks(title="NutriVision AI") as demo:
    with gr.Column(elem_classes="gradio-container"):
        gr.Markdown("# NutriVision AI")
        gr.Markdown('<p class="subtitle">Next-Generation Nutrition Intelligence</p>')

        with gr.Tabs(elem_classes="tab-nav"):
            # ── Tab 1: Home ────────────────────────────────────────────────────────
            with gr.Tab("🏠 OVERVIEW"):
                gr.HTML(HOME_HTML)

            # ── Tab 2: Food Analyzer ──────────────────────────────────────────────
            with gr.Tab("📷 ANALYZE"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):
                        with gr.Column(elem_classes="glass-card"):
                            image_input   = gr.Image(type="pil", label="Capture / Upload Meal")
                            lang_toggle   = gr.Radio(["English", "Tamil"], label="Interface Language", value="English")
                            analyze_btn   = gr.Button("🔍 ANALYZE NOW", variant="primary", elem_classes="primary")
                            pdf_output    = gr.File(label="📄 PDF EXPORT")
                    with gr.Column(scale=2):
                        with gr.Column(elem_classes="glass-card"):
                            annotated_out = gr.Image(type="pil", label="AI VISION OVERLAY")
                        nutrient_display = gr.HTML(label="COMPUTATIONAL METRICS")
                        with gr.Row():
                            with gr.Column(elem_classes="glass-card"):
                                macro_chart = gr.Image(label="MACRO DISTRIBUTION")
                            with gr.Column(elem_classes="glass-card"):
                                cal_chart   = gr.Image(label="CALORIC PROXIMITY")
                
                analyze_btn.click(
                    fn=process_image,
                    inputs=[image_input, lang_toggle],
                    outputs=[annotated_out, nutrient_display, macro_chart, cal_chart, pdf_output]
                )

            # ── Tab 3: Nutrition Database ──────────────────────────────────────────
            with gr.Tab("🗄️ DATABASE"):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### CORE NUTRITION REPOSITORY")
                    with gr.Row():
                        db_search = gr.Textbox(placeholder="Query food item...", label="SEARCH ENGINE", scale=4)
                        db_btn    = gr.Button("EXECUTE", variant="primary", scale=1, elem_classes="primary")
                    db_output = gr.HTML()
                db_btn.click(fn=get_nutrition_db_html, inputs=[db_search], outputs=[db_output])
                demo.load(fn=lambda: get_nutrition_db_html(""), outputs=[db_output])

            # ── Tab 4: Meal Planner ────────────────────────────────────────────────
            with gr.Tab("📅 TRACKER"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="glass-card"):
                        gr.Markdown("### LOG ENTRY")
                        mp_food = gr.Textbox(label="FOOD VECTOR", placeholder="e.g. burger...", scale=1)
                        mp_grams = gr.Number(label="MASS (G)", value=100, minimum=1, scale=1)
                        mp_meal  = gr.Dropdown(["Breakfast","Lunch","Snack","Dinner"], label="PHASE", value="Lunch", scale=1)
                        mp_add   = gr.Button("⚡ COMMIT RECORD", variant="primary", elem_classes="primary")
                        mp_clear = gr.Button("🗑️ PURGE LOG", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Column(elem_classes="glass-card"):
                            mp_log_html    = gr.HTML(label="HISTORICAL VECTOR")
                        with gr.Column(elem_classes="glass-card"):
                            mp_summary_html = gr.HTML(label="CUMULATIVE METRICS")
                
                mp_add.click(fn=log_meal_entry, inputs=[mp_food, mp_grams, mp_meal], outputs=[mp_log_html, mp_summary_html])
                mp_clear.click(fn=clear_meal_log, outputs=[mp_log_html, mp_summary_html])
                demo.load(fn=lambda: (get_meal_log_html(), get_daily_summary_html()), outputs=[mp_log_html, mp_summary_html])

            # ── Tab 5: Diet Insights ───────────────────────────────────────────────
            with gr.Tab("📈 ANALYTICS"):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### COGNITIVE DIET SCORING")
                    refresh_btn = gr.Button("🔄 RECALCULATE INSIGHTS", variant="primary", elem_classes="primary")
                    insights_html = gr.HTML()
                refresh_btn.click(fn=get_diet_insights_html, outputs=[insights_html])
                demo.load(fn=get_diet_insights_html, outputs=[insights_html])

            # ── Tab 6: Meal Plans ──────────────────────────────────────────────────
            with gr.Tab("🍽️ BLUEPRINTS"):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### OPTIMIZATION BLUEPRINTS")
                    plan_goal = gr.Radio(["Weight Loss","Muscle Gain","Balanced Health"], label="TARGET GOAL", value="Balanced Health")
                    plan_out  = gr.HTML()
                plan_goal.change(fn=get_meal_plan_html, inputs=[plan_goal], outputs=[plan_out])
                demo.load(fn=lambda: get_meal_plan_html("Balanced Health"), outputs=[plan_out])

            # ── Tab 7: About ───────────────────────────────────────────────────────
            with gr.Tab("ℹ️ ARCHITECTURE"):
                gr.HTML(ABOUT_HTML)

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    demo.launch(server_port=port, share=False, theme=gr.themes.Base(), css=custom_css)
