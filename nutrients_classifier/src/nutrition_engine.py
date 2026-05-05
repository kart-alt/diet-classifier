import json
import pandas as pd
from pathlib import Path
from fuzzywuzzy import process

class NutritionEngine:
    def __init__(self, data_dir=None):
        if data_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.data_dir = base_dir / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.mapping_file = self.data_dir / "processed" / "nutrition_mapping.json"
        self.csv_file = self.data_dir / "raw" / "food_nutrition.csv"
        
        self.exact_mapping = {}
        self.fallback_db = pd.DataFrame()
        self.fallback_foods = []
        self.fallback_dict = {}
        
        self.load_databases()

    def load_databases(self):
        # Load exact O(1) mapping created during preprocessing
        if self.mapping_file.exists():
            with open(self.mapping_file) as f:
                self.exact_mapping = json.load(f)
                
        # Load fallback CSV
        if self.csv_file.exists():
            self.fallback_db = pd.read_csv(self.csv_file)
            if 'Food' in self.fallback_db.columns:
                self.fallback_db = self.fallback_db.drop_duplicates(subset=['Food'])
                self.fallback_foods = self.fallback_db['Food'].tolist()
                self.fallback_dict = self.fallback_db.set_index('Food').to_dict(orient='index')

    def get_nutrition_per_100g(self, food_name):
        """Returns the nutritional dictionary for 100g of the food."""
        # Special handling for specific beverages
        if food_name.lower() in ['cup', 'beverage', 'drink', 'coca_cola', 'coke', 'cola']:
            # Coca Cola/Cola has ~42 kcal per 100ml
            return {
                "Calories (kcal)": 42,
                "Protein (g)": 0.0,
                "Fat (g)": 0.0,
                "Carbs (g)": 10.5,
                "Fiber (g)": 0.0,
                "Sugar (g)": 10.0,
                "Sodium (mg)": 30.0,
                "Vitamin C (mg)": 0.0,
                "Calcium (mg)": 5.0,
                "Iron (mg)": 0.0
            }
        
        # 1. Check Exact Mapping first
        if food_name in self.exact_mapping:
            entry = self.exact_mapping[food_name]
            if entry.get('match_name') != 'Unknown':
                # Map standardized keys
                return {
                    "Calories (kcal)": entry.get("calories", 0),
                    "Protein (g)": entry.get("protein", 0),
                    "Fat (g)": entry.get("fat", 0),
                    "Carbs (g)": entry.get("carbs", 0),
                    "Fiber (g)": entry.get("fiber", 0),
                    "Sugar (g)": entry.get("sugar", 0),
                    "Sodium (mg)": entry.get("sodium", 0),
                    "Vitamin C (mg)": entry.get("vit_c", 0),
                    "Calcium (mg)": entry.get("calcium", 0),
                    "Iron (mg)": entry.get("iron", 0)
                }
                
        # 2. Check Fallback Fuzzy Matching If Not In Exact Mapping
        if self.fallback_foods:
            best_match, score = process.extractOne(food_name.replace("_", " "), self.fallback_foods)
            if score >= 60:
                nutrients = self.fallback_dict[best_match]
                return {
                    "Calories (kcal)": nutrients.get("Calories (kcal)", 0),
                    "Protein (g)": nutrients.get("Protein (g)", 0),
                    "Fat (g)": nutrients.get("Fat (g)", 0),
                    "Carbs (g)": nutrients.get("Carbs (g)", 0),
                    "Fiber (g)": nutrients.get("Fiber (g)", 0),
                    "Sugar (g)": nutrients.get("Sugar (g)", 0),
                    "Sodium (mg)": nutrients.get("Sodium (mg)", 0),
                    "Vitamin C (mg)": nutrients.get("Vitamin C (mg)", 0),
                    "Calcium (mg)": nutrients.get("Calcium (mg)", 0),
                    "Iron (mg)": nutrients.get("Iron (mg)", 0)
                }
                
        # 3. Last resort if not found anywhere
        raise ValueError(f"Food item '{food_name}' could not be matched in the nutrition database.")

    def compute_plate_nutrition(self, detections):
        """
        Computes total nutrition for a list of detected foods and their weights.
        detections: list of dicts [{'food': 'pizza', 'grams': 150}, ...]
        """
        total_nutrition = {
            "Calories (kcal)": 0, "Protein (g)": 0, "Fat (g)": 0, "Carbs (g)": 0,
            "Fiber (g)": 0, "Sugar (g)": 0, "Sodium (mg)": 0, "Vitamin C (mg)": 0,
            "Calcium (mg)": 0, "Iron (mg)": 0
        }
        
        breakdown = []
        
        for item in detections:
            food_name = item['food']
            grams = item['grams']
            scale = grams / 100.0
            
            try:
                nutrients_100g = self.get_nutrition_per_100g(food_name)
                scaled_nutrients = {k: v * scale for k, v in nutrients_100g.items() if isinstance(v, (int, float))}
                
                # Accumulate
                for k in total_nutrition.keys():
                    total_nutrition[k] += scaled_nutrients.get(k, 0)
                    
                breakdown.append({
                    "food": food_name,
                    "grams": grams,
                    "nutrients": scaled_nutrients
                })
            except ValueError as e:
                print(f"Warning: {e}")
                breakdown.append({
                    "food": food_name,
                    "grams": grams,
                    "error": str(e)
                })
                
        return {"total": total_nutrition, "breakdown": breakdown}

    def format_nutrition_label(self, total_nutrition):
        """Returns a string formatted like a classic Nutrition Facts label."""
        label = f"""
======================================
           NUTRITION FACTS            
======================================
Amount Per Serving
--------------------------------------
Calories:                 {total_nutrition['Calories (kcal)']:.0f} kcal
--------------------------------------
Total Fat                 {total_nutrition['Fat (g)']:.1f} g
Total Carbohydrates       {total_nutrition['Carbs (g)']:.1f} g
  Dietary Fiber           {total_nutrition['Fiber (g)']:.1f} g
  Total Sugars            {total_nutrition['Sugar (g)']:.1f} g
Protein                   {total_nutrition['Protein (g)']:.1f} g
--------------------------------------
Sodium                    {total_nutrition['Sodium (mg)']:.0f} mg
Vitamin C                 {total_nutrition['Vitamin C (mg)']:.1f} mg
Calcium                   {total_nutrition['Calcium (mg)']:.1f} mg
Iron                      {total_nutrition['Iron (mg)']:.1f} mg
======================================
"""
        return label
