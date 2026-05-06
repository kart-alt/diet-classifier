import os
import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.nutrition_engine import NutritionEngine

# ─── Global State ─────────────────────────────────────────────────────────────
print("Initializing FastAPI Server and Loading Models...")
base_dir = Path(__file__).resolve().parent
classifier_path       = base_dir / "models" / "food_classifier_custom.h5"
portion_path          = base_dir / "models" / "portion_estimator.h5"
label_map_path        = base_dir / "data" / "processed" / "label_map.json"
yolo_mapping_path     = base_dir / "data" / "processed" / "yolo_to_nutrition_mapping.json"
indian_model_path     = base_dir / "models" / "indian_food_model.h5"
indian_label_map_path = base_dir / "data" / "processed" / "indian_label_map.json"

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
    print("Custom Classifier (Food-101) loaded.")
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
    label_map = {k: v for k, v in raw_map.items() if v != "Food Classification"}
    print(f"Food-101 label map loaded: {len(label_map)} classes.")
except Exception:
    label_map = {str(i): f"Food_{i}" for i in range(101)}

# Load Indian Food model (trained via train_custom_model.py)
try:
    indian_classifier = tf.keras.models.load_model(str(indian_model_path))
    with open(indian_label_map_path) as f:
        indian_label_map = json.load(f)
    print(f"Indian Food Classifier loaded: {len(indian_label_map)} classes.")
except Exception as e:
    print(f"Indian Food model not found (run train_custom_model.py to generate it): {e}")
    indian_classifier = None
    indian_label_map = {}

try:
    from ultralytics import YOLO
    object_detector = YOLO(str(base_dir / "yolov8n.pt"))
    print("YOLOv8 model loaded.")
except Exception as e:
    print(f"Could not load YOLO: {e}.")
    object_detector = None

IMG_SIZE = 224

# ─── Setup FastAPI ─────────────────────────────────────────────────────────────
app = FastAPI(title="NutriVision API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def classify_crop(crop_img, area_ratio=None):
    img_resized = crop_img.resize((IMG_SIZE, IMG_SIZE))
    img_array   = np.array(img_resized) / 255.0
    img_tensor  = np.expand_dims(img_array, axis=0)

    best_name = "Unknown"
    best_conf = 0.0

    # --- Run Food-101 CNN ---
    if classifier is not None:
        preds = classifier.predict(img_tensor, verbose=0)[0]
        max_prob = float(preds.max())
        if max_prob >= 0.02:
            top_idx = preds.argmax()
            name = label_map.get(str(top_idx), f"Food_{top_idx}").replace("_", " ").title()
            conf = max_prob * 100
            if conf > best_conf:
                best_name = name
                best_conf = conf

    # --- Run Indian Food CNN ---
    if indian_classifier is not None:
        preds_i = indian_classifier.predict(img_tensor, verbose=0)[0]
        max_prob_i = float(preds_i.max())
        if max_prob_i >= 0.02:
            top_idx_i = preds_i.argmax()
            name_i = indian_label_map.get(str(top_idx_i), f"Food_{top_idx_i}").replace("_", " ").title()
            conf_i = max_prob_i * 100
            if conf_i > best_conf:
                best_name = name_i
                best_conf = conf_i

    if best_name == "Unknown":
        return "Unknown", 0.0, 200.0

    if portion_estimator is not None:
        est_grams = max(50.0, float(portion_estimator.predict(img_tensor, verbose=0)[0][0]))
    else:
        est_grams = max(50.0, float(area_ratio) * 700.0) if area_ratio is not None else 200.0

    return best_name, best_conf, est_grams


def nms_boxes(boxes, overlap_threshold=0.5):
    if not boxes: return []
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
    detections = []
    w, h = image.size
    cw, ch = w // grid_size, h // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            x1, y1, x2, y2 = j * cw, i * ch, j * cw + cw, i * ch + ch
            crop = image.crop((x1, y1, x2, y2))
            food_name, conf, _ = classify_crop(crop)
            if food_name != "Unknown" and conf > 0.5:
                detections.append({'food': food_name, 'box': (x1, y1, x2, y2), 'conf': conf, 'source': 'grid'})
    return detections

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

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
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

    non_food = {'fork', 'knife', 'spoon', 'plate', 'bowl', 'dining table', 'person', 'potted plant', 'motorcycle', 'bicycle', 'car', 'bus', 'train', 'dog', 'cat', 'bird', 'mouse', 'chair', 'couch'}

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        if x2 <= x1 or y2 <= y1: continue
        yc    = yolo_results[idx]['class'] if idx < len(yolo_results) else None
        yconf = yolo_results[idx]['conf']  if idx < len(yolo_results) else 0.0

        if yc and yc.lower() in non_food: continue

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
            detections_res.append({"food": food_name, "conf": conf, "grams": est_grams, "box": (x1, y1, x2, y2)})

    for gd in grid_detections:
        if gd['food'] != "Unknown" and gd['conf'] > 0.5:
            gb = gd['box']
            is_dup = any(gb[2] > e['box'][0] and gb[0] < e['box'][2] and gb[3] > e['box'][1] and gb[1] < e['box'][3] for e in detections_res)
            if not is_dup:
                ga = (gb[2] - gb[0]) * (gb[3] - gb[1])
                ia = image.size[0] * image.size[1]
                detections_res.append({"food": gd['food'], "conf": gd['conf'], "grams": max(50.0, ga / ia * 700.0), "box": gb})

    compute_list    = [{"food": d["food"], "grams": d["grams"]} for d in detections_res]
    result          = engine.compute_plate_nutrition(compute_list)
    total_nutrients = result["total"]

    # Build response data
    response_detections = []
    insights = []
    shown_foods = set()

    for d in detections_res:
        try:
            cals = (d['grams'] / 100.0) * engine.get_nutrition_per_100g(d['food'].lower()).get("Calories (kcal)", 0)
        except Exception:
            cals = 0
            
        emoji = get_emoji_for_food(d['food'])
        response_detections.append({
            "food": d['food'],
            "grams": float(d['grams']),
            "conf": float(d['conf']),
            "box": d['box'],
            "calories": float(cals),
            "emoji": emoji
        })

        food_key = d["food"].lower().replace(" ", "_")
        if food_key in FOOD_TIPS and food_key not in shown_foods:
            shown_foods.add(food_key)
            insight = {"title": f"Tip for {d['food']}", "desc": FOOD_TIPS[food_key], "icon": "💡", "type": "tip"}
            insights.append(insight)
            if food_key in HEALTHY_ALTERNATIVES:
                alts = ", ".join(HEALTHY_ALTERNATIVES[food_key])
                insights.append({"title": "Healthier Alternatives", "desc": alts, "icon": "🥦", "type": "alt"})

    return JSONResponse(content={
        "detections": response_detections,
        "nutrients": {
            "Calories (kcal)": float(total_nutrients.get("Calories (kcal)", 0)),
            "Protein (g)": float(total_nutrients.get("Protein (g)", 0)),
            "Carbs (g)": float(total_nutrients.get("Carbs (g)", 0)),
            "Fat (g)": float(total_nutrients.get("Fat (g)", 0)),
            "Fiber (g)": float(total_nutrients.get("Fiber (g)", 0)),
            "Sodium (mg)": float(total_nutrients.get("Sodium (mg)", 0)),
            "Vitamin C (mg)": float(total_nutrients.get("Vitamin C (mg)", 0)),
            "Calcium (mg)": float(total_nutrients.get("Calcium (mg)", 0)),
            "Iron (mg)": float(total_nutrients.get("Iron (mg)", 0)),
            "Sugar (g)": float(total_nutrients.get("Sugar (g)", 0))
        },
        "insights": insights,
        "image_size": [image.size[0], image.size[1]]
    })

from database import engine as db_engine, Base, get_db
from sqlalchemy.orm import Session
from fastapi import Depends
import models
import datetime

# Create database tables
models.Base.metadata.create_all(bind=db_engine)

@app.get("/api/database/search")
def search_database(q: str = ""):
    results = engine.search_database(q, limit=50)
    return {"results": results}

from pydantic import BaseModel

import hashlib
from fastapi import HTTPException

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

class UserCreate(BaseModel):
    username: str
    password: str

@app.post("/api/auth/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_user = models.User(username=user.username, password_hash=hash_password(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"status": "success", "user_id": new_user.id, "username": new_user.username}

@app.post("/api/auth/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or db_user.password_hash != hash_password(user.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    return {"status": "success", "user_id": db_user.id, "username": db_user.username}

class MealCreate(BaseModel):
    user_id: int
    meal_type: str
    food_name: str
    grams: float
    calories: float
    protein: float
    carbs: float
    fat: float

@app.post("/api/meals")
def log_meal(meal: MealCreate, db: Session = Depends(get_db)):
    db_meal = models.MealLog(**meal.dict())
    db.add(db_meal)
    db.commit()
    db.refresh(db_meal)
    return {"status": "success", "meal": db_meal}

@app.get("/api/meals")
def get_meals(user_id: int, db: Session = Depends(get_db)):
    # Get today's meals
    today = datetime.datetime.utcnow().date()
    meals = db.query(models.MealLog).filter(models.MealLog.user_id == user_id, models.MealLog.timestamp >= today).all()
    
    # Also calculate 7-day history for analytics
    seven_days_ago = today - datetime.timedelta(days=7)
    history = db.query(models.MealLog).filter(models.MealLog.user_id == user_id, models.MealLog.timestamp >= seven_days_ago).all()
    
    return {"today": meals, "history": history}

import google.generativeai as genai
import os

# Initialize Gemini client
# Set your API key here or via environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAknbSDPFJ40GlKrvvaV7fTIDLQDBMbEoc")
genai.configure(api_key=GOOGLE_API_KEY)
# Use the fast and efficient Gemini 2.5 Flash model
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

class ChatRequest(BaseModel):
    user_id: int
    message: str

@app.post("/api/chat")
def chat_with_nutritionist(req: ChatRequest, db: Session = Depends(get_db)):
    try:
        # Get today's meals for context
        today = datetime.datetime.utcnow().date()
        meals = db.query(models.MealLog).filter(models.MealLog.user_id == req.user_id, models.MealLog.timestamp >= today).all()
        
        meal_context = "User has not eaten anything today."
        if meals:
            total_cals = sum(m.calories for m in meals if m.calories)
            total_pro = sum(m.protein for m in meals if m.protein)
            meal_list = ", ".join([f"{m.food_name} ({m.grams}g)" for m in meals])
            meal_context = f"Today the user ate: {meal_list}. Total calories so far: {total_cals:.0f} kcal, Total protein: {total_pro:.0f}g."

        system_prompt = f"""You are NutriVision AI, an expert, friendly conversational nutritionist. 
Your goal is to provide concise, accurate, and highly actionable dietary advice.
Current User Context: {meal_context}
Keep responses under 3 sentences unless asked for a detailed breakdown."""

        # Combine system prompt with user message since Gemini API uses history or a combined prompt
        prompt = f"{system_prompt}\n\nUser Question: {req.message}"
        
        response = gemini_model.generate_content(prompt)
        
        return {"reply": response.text}
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"reply": f"I'm having trouble connecting to Gemini: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
