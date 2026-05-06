import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import io
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ─── Tamil translations ────────────────────────────────────────────────────────
TAMIL_MAPPING = {
    "Calories (kcal)":    "கலோரிகள் (kcal)",
    "Protein (g)":        "புரதம் (g)",
    "Fat (g)":            "கொழுப்பு (g)",
    "Carbs (g)":          "மாவுச்சத்து (g)",
    "Fiber (g)":          "நார்ச்சத்து (g)",
    "Sugar (g)":          "சர்க்கரை (g)",
    "Sodium (mg)":        "சோடியம் (mg)",
    "Vitamin C (mg)":     "வைட்டமின் சி (mg)",
    "Calcium (mg)":       "கால்சியம் (mg)",
    "Iron (mg)":          "இரும்புச்சத்து (mg)",
    "Amount Per Serving": "ஒரு சேவைக்கு அளவு",
    "Nutrition Facts":    "ஊட்டச்சத்து உண்மைகள்",
    "Total Fat":          "மொத்த கொழுப்பு",
    "Total Carbohydrates":"மொத்த மாவுச்சத்து",
    "Total Sugars":       "மொத்த சர்க்கரை",
    "Detected Items":     "கண்டறியப்பட்ட உணவுகள்",
    "Grams":              "கிராம்",
    "Macronutrients":     "பேரூட்டச்சத்துக்கள்",
}


def translate_to_tamil(text: str) -> str:
    """Replace English nutrition keys with Tamil equivalents."""
    for eng, tam in TAMIL_MAPPING.items():
        text = text.replace(eng, tam)
    return text


# ─── Chart helpers ─────────────────────────────────────────────────────────────

def _dark_style():
    """Apply a consistent dark theme to the current figure."""
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor":   "#111111",
        "axes.edgecolor":   "#333333",
        "axes.labelcolor":  "#cccccc",
        "xtick.color":      "#cccccc",
        "ytick.color":      "#cccccc",
        "text.color":       "#ffffff",
        "grid.color":       "#222222",
    })


def plot_macronutrients(nutrition_data: dict, lang: str = "English") -> io.BytesIO:
    """Horizontal bar chart of the three main macros. Returns a PNG BytesIO."""
    _dark_style()

    macros = ["Protein (g)", "Carbs (g)", "Fat (g)"]
    values = [nutrition_data.get(m, 0) for m in macros]

    if lang == "Tamil":
        labels = [TAMIL_MAPPING.get(m, m) for m in macros]
        x_label = TAMIL_MAPPING.get("Grams", "Grams")
        title   = TAMIL_MAPPING.get("Macronutrients", "Macronutrients")
    else:
        labels  = macros
        x_label = "Grams"
        title   = "Macronutrients"

    colors = ["#2980b9", "#8e44ad", "#f39c12"]

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0a0a0a")
    ax.set_facecolor("#111111")
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.5)

    ax.set_xlabel(x_label, color="#cccccc")
    ax.set_title(title, color="#ffffff", fontweight="bold", pad=10)
    ax.tick_params(colors="#cccccc")
    ax.spines[:].set_color("#333333")

    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}g", va="center", ha="left", color="#ffffff", fontsize=9,
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor="#0a0a0a")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_calorie_distribution(nutrition_data: dict, lang: str = "English") -> io.BytesIO:
    """Pie chart of calories from protein / carbs / fat. Returns a PNG BytesIO."""
    _dark_style()

    p_cal = nutrition_data.get("Protein (g)", 0) * 4
    c_cal = nutrition_data.get("Carbs (g)",   0) * 4
    f_cal = nutrition_data.get("Fat (g)",     0) * 9

    if lang == "Tamil":
        labels = ["புரதம்", "மாவுச்சத்து", "கொழுப்பு"]
        title  = "கலோரி பகிர்வு"
    else:
        labels = ["Protein", "Carbs", "Fat"]
        title  = "Calorie Distribution"

    sizes  = [p_cal, c_cal, f_cal]
    colors = ["#2980b9", "#8e44ad", "#f39c12"]

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    if sum(sizes) > 0:
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "#ffffff"},
            wedgeprops={"edgecolor": "#0a0a0a", "linewidth": 2},
        )
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_fontsize(9)
    else:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=12)

    ax.set_title(title, color="#ffffff", fontweight="bold", pad=12)
    ax.axis("equal")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor="#0a0a0a")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── PDF generation ────────────────────────────────────────────────────────────

def generate_pdf_report(
    nutrition_text: str,
    image_path,                        # currently unused; reserved for future image embed
    output_filename: str = "nutrition_report.pdf",
) -> str:
    """
    Write a plain-text Nutrition Facts report to a PDF.
    Falls back to Helvetica/Courier when the Tamil font is absent.
    Returns the absolute path to the saved PDF.
    """
    from reportlab.pdfbase        import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    out_path = Path(output_filename).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_dir   = Path(__file__).resolve().parent.parent
    font_path  = base_dir / "assets" / "fonts" / "NotoSansTamil-Regular.ttf"

    # ── Font selection ──────────────────────────────────────────────────────────
    if font_path.exists():
        try:
            pdfmetrics.registerFont(TTFont("NotoSansTamil", str(font_path)))
            title_font = "NotoSansTamil"
            body_font  = "NotoSansTamil"
        except Exception:
            title_font = "Helvetica-Bold"
            body_font  = "Courier"
    else:
        title_font = "Helvetica-Bold"
        body_font  = "Courier"

    # ── Build PDF ───────────────────────────────────────────────────────────────
    c      = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    margin = 50
    y_pos  = height - margin

    # Title
    c.setFont(title_font, 18)
    c.setFillColorRGB(0.08, 0.08, 0.08)
    c.rect(0, height - 70, width, 70, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.drawString(margin, height - 45, "NutriVision AI — Nutrition Report")

    # Timestamp
    import datetime
    c.setFont(body_font, 8)
    c.setFillColorRGB(0.7, 0.7, 0.7)
    c.drawRightString(
        width - margin, height - 55,
        datetime.datetime.now().strftime("Generated: %Y-%m-%d %H:%M"),
    )

    y_pos = height - 90
    c.setFont(body_font, 11)
    c.setFillColorRGB(0.1, 0.1, 0.1)

    # Body lines — start a new page if we run out of space
    for line in nutrition_text.split("\n"):
        if y_pos < 60:
            c.showPage()
            y_pos = height - margin
            c.setFont(body_font, 11)
            c.setFillColorRGB(0.1, 0.1, 0.1)

        # Use a slightly larger font for section headers (lines of "=")
        if line.strip().startswith("=") or line.strip().startswith("-"):
            c.setFillColorRGB(0.4, 0.4, 0.4)
        else:
            c.setFillColorRGB(0.1, 0.1, 0.1)

        c.drawString(margin, y_pos, line.strip()[:95])   # truncate very long lines
        y_pos -= 16

    c.save()
    return str(out_path)