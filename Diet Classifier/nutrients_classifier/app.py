import os
import json
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

from src.nutrition_engine import NutritionEngine
from src.utils import translate_to_tamil, plot_macronutrients, plot_calorie_distribution, generate_pdf_report

# ─── Global State: Load Models Once at Startup ────────────────────────────────
print("Initializing App State and Loading Models...")
base_dir = Path(__file__).resolve().parent
classifier_path = base_dir / "models" / "food_classifier_custom.h5"
portion_path    = base_dir / "models" / "portion_estimator.h5"
label_map_path  = base_dir / "data" / "processed" / "label_map.json"

engine = NutritionEngine(base_dir / "data")

try:
    classifier = tf.keras.models.load_model(str(classifier_path))
    print("✅ Custom Classifier loaded.")
except Exception as e:
    print(f"⚠️  Classifier not found: {e}. Will use MobileNetV2 fallback.")
    classifier = None

try:
    portion_estimator = tf.keras.models.load_model(str(portion_path))
    print("✅ Portion Estimator loaded.")
except Exception as e:
    print("⚠️  Portion Estimator not found. Using 200g fallback.")
    portion_estimator = None

try:
    with open(label_map_path) as f:
        label_map = json.load(f)
except Exception:
    print("⚠️  label_map.json not found.")
    label_map = {str(i): f"Food_{i}" for i in range(101)}

# Pre-load MobileNetV2 fallback once (only when no custom classifier)
fallback_model         = None
_mobilenet_preprocess  = None
_mobilenet_decode      = None

if classifier is None:
    print("Loading MobileNetV2 fallback model (one-time)...")
    try:
        from tensorflow.keras.applications.mobilenet_v2 import (
            MobileNetV2, preprocess_input, decode_predictions
        )
        fallback_model        = MobileNetV2(weights="imagenet")
        _mobilenet_preprocess = preprocess_input
        _mobilenet_decode     = decode_predictions
        print("✅ MobileNetV2 fallback loaded.")
    except Exception as e:
        print(f"⚠️  Could not load MobileNetV2: {e}")

IMG_SIZE = 224

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _level_bar(value, max_val, color):
    """Return a small HTML progress bar."""
    pct = min(100, int((value / max_val) * 100)) if max_val else 0
    return (
        f'<div style="background:#e8e8e8;border-radius:6px;height:10px;width:100%;">'
        f'<div style="width:{pct}%;background:{color};border-radius:6px;height:10px;"></div>'
        f'</div>'
    )

def _nutrient_level_label(value, thresholds):
    """Return Low / Medium / High label colour."""
    if value <= thresholds[0]:
        return "Low", "#27ae60"
    elif value <= thresholds[1]:
        return "Medium", "#f39c12"
    else:
        return "High", "#e74c3c"

# Daily reference values (for bar scaling)
DRI = {
    "Calories (kcal)": 2000,
    "Protein (g)": 50,
    "Fat (g)": 70,
    "Carbs (g)": 300,
    "Fiber (g)": 28,
    "Sugar (g)": 50,
    "Sodium (mg)": 2300,
    "Vitamin C (mg)": 90,
    "Calcium (mg)": 1000,
    "Iron (mg)": 18,
}

NUTRIENT_META = {
    # key: (display_name, unit, bar_color, (low_thresh, high_thresh))
    "Calories (kcal)": ("🔥 Calories",     "kcal", "#e74c3c", (300, 700)),
    "Protein (g)":     ("💪 Protein",      "g",    "#2980b9", (10,  25)),
    "Fat (g)":         ("🥑 Fat",          "g",    "#f39c12", (10,  25)),
    "Carbs (g)":       ("🍞 Carbohydrates","g",    "#8e44ad", (30,  80)),
    "Fiber (g)":       ("🌿 Fiber",        "g",    "#27ae60", (5,   15)),
    "Sugar (g)":       ("🍬 Sugar",        "g",    "#e67e22", (5,   20)),
    "Sodium (mg)":     ("🧂 Sodium",       "mg",   "#95a5a6", (300, 800)),
    "Vitamin C (mg)":  ("🍊 Vitamin C",    "mg",   "#f1c40f", (15,  60)),
    "Calcium (mg)":    ("🦴 Calcium",      "mg",   "#1abc9c", (200, 600)),
    "Iron (mg)":       ("⚙️ Iron",         "mg",   "#7f8c8d", (3,   10)),
}

def build_nutrient_html(nutrients, food_name, est_grams, top_predictions):
    """Build a rich HTML card showing food classification + nutrient levels."""

    # ── Top predictions block ──────────────────────────────────────────
    preds_html = ""
    for i, (name, conf) in enumerate(top_predictions):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        bar_w = int(conf)
        preds_html += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <span style="font-size:1.3rem;">{medal}</span>
          <div style="flex:1;">
            <div style="display:flex;justify-content:space-between;">
              <span style="font-weight:600;color:#2c3e50;">{name}</span>
              <span style="font-weight:700;color:#667eea;">{conf:.1f}%</span>
            </div>
            <div style="background:#eee;border-radius:6px;height:8px;margin-top:4px;">
              <div style="width:{bar_w}%;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:6px;height:8px;"></div>
            </div>
          </div>
        </div>"""

    # ── Nutrient rows ──────────────────────────────────────────────────
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

    html = f"""
<div style="font-family:'Inter',sans-serif;max-width:720px;">

  <!-- Food Detection Card -->
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);border-radius:16px;padding:20px 24px;margin-bottom:16px;color:white;">
    <div style="font-size:0.85rem;opacity:0.85;margin-bottom:4px;">🍽️ TOP CLASSIFICATION RESULT</div>
    <div style="font-size:1.8rem;font-weight:800;letter-spacing:-0.5px;">{food_name}</div>
    <div style="margin-top:8px;font-size:1rem;opacity:0.9;">
      ⚖️ Estimated Portion: <strong>{est_grams:.0f} g</strong>
    </div>
  </div>

  <!-- Top Predictions -->
  <div style="background:#fff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 20px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <div style="font-weight:700;font-size:1rem;color:#2c3e50;margin-bottom:14px;">📊 Food Classification Rankings</div>
    {preds_html}
  </div>

  <!-- Nutrient Breakdown -->
  <div style="background:#fff;border:1px solid #e8e8e8;border-radius:14px;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <div style="font-weight:700;font-size:1rem;color:#2c3e50;margin-bottom:12px;">🧬 Nutrient Levels <span style="font-size:0.8rem;color:#888;font-weight:400;">(based on {est_grams:.0f}g serving)</span></div>
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="border-bottom:2px solid #eee;">
          <th style="padding:6px;text-align:left;color:#888;font-size:0.8rem;font-weight:600;">NUTRIENT</th>
          <th style="padding:6px;text-align:right;color:#888;font-size:0.8rem;font-weight:600;">AMOUNT</th>
          <th style="padding:6px;color:#888;font-size:0.8rem;font-weight:600;">LEVEL</th>
          <th style="padding:6px;text-align:center;color:#888;font-size:0.8rem;font-weight:600;">STATUS</th>
          <th style="padding:6px;text-align:right;color:#888;font-size:0.8rem;font-weight:600;">DAILY %</th>
        </tr>
      </thead>
      <tbody>{nutrient_rows}</tbody>
    </table>
  </div>

</div>
"""
    return html


# ─── Main Processing Function ──────────────────────────────────────────────────

def process_image(image, language):
    if image is None:
        return "Please upload an image first.", None, None, None

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array  = np.array(img_resized) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    top_predictions = []  # list of (name, confidence%)

    # 1. ── Food Classification ───────────────────────────────────────────────
    if classifier is not None:
        preds   = classifier.predict(img_tensor, verbose=0)[0]
        top_idxs = preds.argsort()[::-1][:3]
        for idx in top_idxs:
            name = label_map.get(str(idx), f"Food_{idx}").replace("_", " ").title()
            top_predictions.append((name, float(preds[idx]) * 100))
        food_name = top_predictions[0][0]
        conf      = top_predictions[0][1]
    else:
        if fallback_model is None:
            err = "❌ No model loaded. Please train the classifier or check your setup."
            return err, None, None, None

        img_mn = image.resize((224, 224))
        arr_mn = np.expand_dims(np.array(img_mn), axis=0)
        arr_mn = _mobilenet_preprocess(arr_mn)

        preds   = fallback_model.predict(arr_mn, verbose=0)
        decoded = _mobilenet_decode(preds, top=3)[0]          # top-3

        for _, name, prob in decoded:
            top_predictions.append((name.replace("_", " ").title(), float(prob) * 100))

        food_name = top_predictions[0][0]
        conf      = top_predictions[0][1]

    # 2. ── Portion Estimation ────────────────────────────────────────────────
    if portion_estimator is not None:
        est_grams = float(portion_estimator.predict(img_tensor, verbose=0)[0][0])
        est_grams = max(50.0, est_grams)
    else:
        est_grams = 200.0  # standard fallback

    # 3. ── Nutrition Lookup ──────────────────────────────────────────────────
    detections = [{"food": food_name, "grams": est_grams}]
    result          = engine.compute_plate_nutrition(detections)
    total_nutrients = result["total"]

    # 4. ── Build Outputs ─────────────────────────────────────────────────────
    nutrient_html = build_nutrient_html(total_nutrients, food_name, est_grams, top_predictions)

    from PIL import Image as PILImage
    macro_buf = plot_macronutrients(total_nutrients, language)
    cal_buf   = plot_calorie_distribution(total_nutrients, language)
    macro_img = PILImage.open(macro_buf)
    cal_img   = PILImage.open(cal_buf)

    # PDF report (plain-text summary)
    text_report = (
        f"Food Detected : {food_name}  ({conf:.1f}% confidence)\n"
        f"Estimated Portion: {est_grams:.0f} g\n"
    )
    text_report += engine.format_nutrition_label(total_nutrients)
    if language == "Tamil":
        text_report = translate_to_tamil(text_report)

    out_dir  = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = generate_pdf_report(text_report, None, str(out_dir / "nutrition_report.pdf"))

    return nutrient_html, macro_img, cal_img, pdf_path


# ─── Custom CSS ───────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

body {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}
.gradio-container {
    max-width: 1300px !important;
    background: rgba(255,255,255,0.04);
    border-radius: 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 32px !important;
}
h1 {
    background: linear-gradient(90deg, #667eea, #f64f59);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-weight: 800;
    font-size: 2.4rem !important;
    letter-spacing: -1.5px;
    margin-bottom: 4px;
}
.subtitle {
    text-align: center;
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    margin-bottom: 2rem;
}
.svelte-1ipelgc label span {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 600;
}
.primary-btn button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 14px !important;
    padding: 14px 28px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4);
    letter-spacing: 0.3px;
}
.primary-btn button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(118,75,162,0.5) !important;
}
"""

# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
    gr.Markdown("# 🥗 AI Food Nutrition Classifier")
    gr.Markdown('<p class="subtitle">Upload a meal photo → get instant food classification, portion estimate & nutrient breakdown</p>')

    with gr.Row(equal_height=False):
        # Left: Upload + controls
        with gr.Column(scale=1, min_width=280):
            image_input    = gr.Image(type="pil", label="📷 Upload Food Image")
            language_toggle = gr.Radio(["English", "Tamil"], label="Language", value="English")
            submit_btn     = gr.Button("🔍 Analyze Food & Nutrients", variant="primary", elem_classes="primary-btn")
            pdf_output     = gr.File(label="📄 Download PDF Report")

        # Right: Results
        with gr.Column(scale=2):
            nutrient_display = gr.HTML(label="Classification & Nutrient Levels")
            with gr.Row():
                macro_chart = gr.Image(label="📊 Macronutrients")
                cal_chart   = gr.Image(label="🥧 Calorie Distribution")

    submit_btn.click(
        fn=process_image,
        inputs=[image_input, language_toggle],
        outputs=[nutrient_display, macro_chart, cal_chart, pdf_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
