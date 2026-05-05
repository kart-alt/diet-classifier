# 🍔 Fine-Tuning Guide: Adding Custom Food Classes

Your classifier was trained on fancy dishes (Food-101). To detect **basic foods like french fries, cola, burgers**, you need to fine-tune it.

## 🎯 Quick Start (3 Steps)

### Step 1: Download Training Images
```bash
cd nutrients_classifier
python src/download_additional_foods.py
```

This downloads ~30 images each for:
- French Fries
- Cola/Soda  
- Burger
- Milkshake
- Chips
- Ice Cream
- Donut
- Coffee

**Takes ~2-3 minutes**

### Step 2: Review & Clean Images
```
nutrients_classifier/data/raw/additional_foods/
├── french_fries/     ← Review images, delete bad ones
├── cola/
├── burger/
└── ...
```

Remove:
- Blurry images
- Wrong food items
- Logos/brand images (optional)

**Target: 15-25 good images per category**

### Step 3: Fine-Tune the Model
```bash
python src/finetune_model.py
```

This will:
- Load your pre-trained classifier
- Add new food classes
- Train on additional + Food-101 data
- Save updated model

**Takes ~10-30 minutes** (depending on GPU)

---

## 🔄 Full Workflow

### Option A: Automated Download (Easiest)
```bash
python src/download_additional_foods.py    # 2-3 min
# Review images
python src/finetune_model.py                # 10-30 min
```

### Option B: Manual Image Collection
1. Create folders manually:
   ```
   data/raw/additional_foods/
   ├── french_fries/
   ├── cola/
   └── burger/
   ```

2. Add images (download from Google Images, Unsplash, etc.)

3. Run fine-tuning:
   ```bash
   python src/finetune_model.py
   ```

### Option C: Use Existing Food-101 Classes
If the food you want is already in Food-101, update the mapping:

Edit `data/processed/yolo_to_nutrition_mapping.json`:
```json
{
  "carrot": ["carrot"],
  "apple": ["apple"],
  ...
}
```

---

## 📊 What Gets Fine-Tuned

The script:
1. **Keeps existing knowledge** from Food-101 (fancy dishes)
2. **Adds new classes** for basic foods
3. **Uses low learning rate** (1e-4) to preserve old knowledge
4. **Trains for 50 epochs** with early stopping

```
Old Model: 101 classes (sophisticated dishes)
  ↓
New Model: 101 + N classes (dishes + basic foods)
```

---

## ✅ After Fine-Tuning

Your app will now detect:
- ✅ Existing Food-101 items
- ✅ **NEW: French fries, cola, burger, etc.**

Test it:
```bash
$env:GRADIO_SERVER_PORT="7867"
python app.py
```

Upload your sandwich+fries+cola image and it should now show:
- Sandwich (mapped from classifier)
- French Fries (from fine-tuned class)
- Cup/Cola (mapped to beverage)

---

## 🐛 Troubleshooting

### "No additional food images found"
→ Run `python src/download_additional_foods.py` first

### "Model training is slow"
→ GPU not detected. Check: `nvidia-smi` in terminal

### "Low accuracy after fine-tuning"
→ Add more training images (50+ per class)
→ Images should be clear, well-lit, single food item

### Images still not detected
→ The food might not be in your nutrition database
→ Add it to `yolo_to_nutrition_mapping.json`

---

## 📈 Advanced: Custom Food Categories

Want to add YOUR specific foods? 

1. Create folder:
   ```
   data/raw/additional_foods/your_food_name/
   ```

2. Add 20-50 images

3. Update nutrition mapping:
   ```json
   {
     "your_food_name": ["your_food_name", "fallback_name"]
   }
   ```

4. Run fine-tuning:
   ```bash
   python src/finetune_model.py
   ```

---

## 📝 Files Modified

When you fine-tune, these files are updated:
- `models/food_classifier_custom.h5` → New model with extra classes
- `data/processed/label_map.json` → Extended class list

Keep backups if you want to revert!

---

## 💡 Tips

1. **Quality > Quantity**: 20 good images > 100 blurry ones
2. **Variety matters**: Different angles, lighting, portions
3. **Test incrementally**: Fine-tune 2-3 foods first, then expand
4. **Monitor RAM**: Close other apps if fine-tuning is slow
5. **Use GPU**: 10x faster than CPU. Check CUDA setup.

---

## 🚀 Next Steps

1. Run: `python src/download_additional_foods.py`
2. Check images in `data/raw/additional_foods/`
3. Run: `python src/finetune_model.py`
4. Test: `python app.py` on your food image

**Estimated total time: 20-40 minutes**

Good luck! 🎯
