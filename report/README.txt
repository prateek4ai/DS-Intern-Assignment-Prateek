# DS-Intern-Assignment-Prateek

## Project Overview

This repository contains the end-to-end solution for the Smart Factory Energy Prediction Challenge at SmartManufacture Inc. We built a data pipeline, performed exploratory analysis, engineered features, trained and tuned machine learning models, and provided scripts for batch prediction.

## Repository Structure


DS-Intern-Assignment-Prateek/
├── data/               # Raw and processed data files
│   ├── data.csv        # Original training dataset
│   └── data_cleaned.csv (optional)
├── docs/               # Documentation and feature descriptions
│   └── data_description.md
├── notebooks/          # Jupyter notebooks for analysis
│   ├── EDA.ipynb       # Exploratory Data Analysis
│   └── modeling.ipynb  # Model training, tuning, and interpretation
├── scripts/            # Python scripts for reproducible workflow
│   ├── preprocess.py   # Data loading, cleaning, and feature engineering
│   ├── model.py        # Training and evaluating baseline and advanced models
│   └── predict.py      # Batch prediction on new data
├── models/             # Saved model artifacts
│   ├── RandomForest.pkl
│   └── RandomForest_tuned.pkl
├── outputs/            # Example prediction outputs
├── report/             # Final report and LaTeX source
│   └── summary.md      # Markdown version of the report
├── README.md           # Project overview and instructions
└── requirements.txt    # Python dependencies


## Setup Instructions

1. Clone the repository:

   
   git clone https://github.com/prateek4ai/DS-Intern-Assignment-Prateek.git
   cd DS-Intern-Assignment-Prateek
   
2. Create a virtual environment (optional but recommended):

  
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   .\venv\Scripts\activate     # Windows
 
3. Install dependencies:

  
   pip install -r requirements.txt
  

## Usage

### 1. Exploratory Data Analysis

* Open and run `notebooks/EDA.ipynb` in Jupyter to inspect raw data, visualize distributions, and quantify missingness.

### 2. Feature Engineering & Modeling

* Launch `notebooks/modeling.ipynb` to perform feature engineering, train baseline models, conduct hyperparameter tuning, and interpret results with SHAP.
* Or run scripts directly:

  
  # Train models and save artifacts
  python scripts/model.py
  

### 3. Batch Prediction

* Use `scripts/predict.py` to generate predictions on new CSV files:

 
  mkdir -p outputs
  python scripts/predict.py data/data.csv outputs/predictions.csv
  
* The script applies the same preprocessing and feature pipeline, then saves `predictions.csv` with timestamps and forecasted energy consumption.

## Report

* See `report/summary.md` for the full analysis, results, and recommendations.
* A LaTeX source is also provided for PDF generation.

## Recommendations

* For faster model training, consider using GPU-accelerated libraries (XGBoost, LightGBM with GPU, or RAPIDS cuML).
* Schedule weekly retraining to adapt to seasonal changes.

Author:
Prateek
Made for:
Mechademy
