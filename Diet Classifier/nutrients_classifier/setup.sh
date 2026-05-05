#!/bin/bash
echo "======================================"
echo "    🍏 Food Nutrients Classifier Setup"
echo "======================================"

# 1. Create Python Virtual Environment
echo "Creating Python virtual environment 'venv'..."
python -m venv venv

# 2. Activate Virtual Environment
# Note: Source command is different on Windows vs Linux/Mac.
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# 3. Install Requirements
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Setup directories
mkdir -p data/raw data/processed models outputs logs

# 5. Run Pipelines
echo "======================================"
echo "Downloading Datasets..."
python src/download_datasets.py

echo "======================================"
echo "Preprocessing Data and Mapping Nutrition..."
python src/preprocess.py

echo "======================================"
echo "Training Custom CNN Model..."
# Uncomment the line below to train the model right after setup. 
# Warning: Training from scratch takes several hours on a GPU.
# python src/train_model.py

echo "======================================"
echo "Launching Web App..."
python app.py
