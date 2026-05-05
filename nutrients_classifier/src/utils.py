import matplotlib.pyplot as plt
import io
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

TAMIL_MAPPING = {
    "Calories (kcal)": "கலோரிகள் (kcal)",
    "Protein (g)": "புரதம் (g)",
    "Fat (g)": "கொழுப்பு (g)",
    "Carbs (g)": "மாவுச்சத்து (g)",
    "Fiber (g)": "நார்ச்சத்து (g)",
    "Sugar (g)": "சர்க்கரை (g)",
    "Sodium (mg)": "சோடியம் (mg)",
    "Vitamin C (mg)": "வைட்டமின் சி (mg)",
    "Calcium (mg)": "கால்சியம் (mg)",
    "Iron (mg)": "இரும்புச்சத்து (mg)",
    "Amount Per Serving": "ஒரு சேவைக்கு அளவு",
    "Nutrition Facts": "ஊட்டச்சத்து உண்மைகள்",
    "Total Fat": "மொத்த கொழுப்பு",
    "Total Carbohydrates": "மொத்த மாவுச்சத்து",
    "Total Sugars": "மொத்த சர்க்கரை"
}

def translate_to_tamil(text):
    for eng, tam in TAMIL_MAPPING.items():
        if eng in text:
            # We preserve standard string templating padding if needed, but exact replace is better
            text = text.replace(eng, tam)
    return text

def plot_macronutrients(nutrition_data, lang="English"):
    macros = ['Protein (g)', 'Carbs (g)', 'Fat (g)']
    values = [nutrition_data.get(m, 0) for m in macros]
    
    labels = macros if lang == "English" else [TAMIL_MAPPING[m] for m in macros]
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel('Grams' if lang == "English" else 'கிராம்')
    ax.set_title('Macronutrients' if lang == "English" else 'பேரூட்டச்சத்துக்கள்')
    
    for i, v in enumerate(values):
         ax.text(v + 0.5, i, f"{v:.1f}g", color='black', va='center')
         
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_calorie_distribution(nutrition_data, lang="English"):
    """Estimates calories from macros: protein: 4, carbs: 4, fat: 9"""
    p_cal = nutrition_data.get('Protein (g)', 0) * 4
    c_cal = nutrition_data.get('Carbs (g)', 0) * 4
    f_cal = nutrition_data.get('Fat (g)', 0) * 9
    
    labels = ['Protein', 'Carbs', 'Fat'] if lang == "English" else ['புரதம்', 'மாவுச்சத்து', 'கொழுப்பு']
    sizes = [p_cal, c_cal, f_cal]
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    fig, ax = plt.subplots(figsize=(4, 4))
    if sum(sizes) > 0:
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    else:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        
    ax.axis('equal')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_pdf_report(nutrition_text, image_path, output_filename="nutrition_report.pdf"):
    """Generates a PDF summary of the nutrition report."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    out_path = Path(output_filename).absolute()
    base_dir = Path(__file__).resolve().parent.parent
    font_path = base_dir / "assets" / "fonts" / "NotoSansTamil-Regular.ttf"
    
    has_tamil_font = font_path.exists()
    if has_tamil_font:
        pdfmetrics.registerFont(TTFont('NotoSansTamil', str(font_path)))
        title_font = 'NotoSansTamil'
        body_font = 'NotoSansTamil'
    else:
        title_font = 'Helvetica-Bold'
        body_font = 'Courier'

    c = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    
    c.setFont(title_font, 20)
    c.drawString(50, height - 50, "Diet Classifier - Nutrition Report" if not has_tamil_font else translate_to_tamil("Diet Classifier - Nutrition Report"))
    
    c.setFont(body_font, 12)
    y_pos = height - 100
    for line in nutrition_text.split('\n'):
        c.drawString(50, y_pos, line.strip())
        y_pos -= 15
        
    c.save()
    return str(out_path)
