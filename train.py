"""
train.py
--------
Train final best model and save it to disk
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

import time
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)

# Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import shap
import joblib

# Set seed
np.random.seed(42)

# Plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("All libraries imported successfully")

# =========================
# 1. Load data
# =========================
MODEL_PATH = "models/footware_model.pkl"

df = pd.read_csv('./data/global_sports_footwear_sales_2018_2026.csv')

# =========================
# 2. Preprocessing
# =========================

# Make a copy
data = df.copy()

# # Remove rows with missing target
# if data['final_price_usd'].isnull().sum() > 0:
#     print(f"Removing {data['final_price_usd'].isnull().sum()} rows with missing target")
#     data = data[~data['final_price_usd'].isnull()]

# Separate target and features
y = data['revenue_usd']

# Remove features that contain target information
leaked_features = [
    'final_price_usd',  # Target itself
    'revenue_usd',      # This is final_price * units_sold - LEAKAGE!
    'order_id',         # Non-informative
    'discount_percent'
]

# Only keep legitimate features
X = data.drop(columns=leaked_features)

# Extract date features
if 'order_date' in X.columns:
    print("\nExtracting date features...")
    X['order_date'] = pd.to_datetime(X['order_date'])
    X['order_year'] = X['order_date'].dt.year
    X['order_month'] = X['order_date'].dt.month
    X['order_day'] = X['order_date'].dt.day
    X['order_dayofweek'] = X['order_date'].dt.dayofweek
    X['order_quarter'] = X['order_date'].dt.quarter
    X = X.drop(columns=['order_date'])

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].fillna('missing')
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} categories")

# Fill missing numerical values
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        print(f"  Filled '{col}' with median: {median_val}")

print(f"\n✓ Preprocessing complete")
print(f"  Final shape: {X.shape}")
print(f"  No missing values: {X.isnull().sum().sum() == 0}")


# =========================
# 3. Train-test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# =========================
# 4. Training Model
# =========================

# Define regressors
regressors = [
    ['Ridge', Ridge(random_state=42)],
    ['Lasso', Lasso(random_state=42, max_iter=5000)],
    ['ElasticNet', ElasticNet(random_state=42, max_iter=5000)],
    ['GradientBoosting', GradientBoostingRegressor(random_state=42)],
    ['XGBoost', XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)],
    ['LightGBM', LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)],
    ['CatBoost', CatBoostRegressor(random_state=42, verbose=0)]
]

model_results = []
trained_models = {}

print("Training models...\n")

for name, model in regressors:
    print(f"Training {name}...", end=' ')
    start_time = time.time()
    
    try:
        # Train
        model.fit(X_train, y_train)
        runtime = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Store
        model_results.append({
            "model_name": name,
            "model": model,
            "runtime": runtime,
            "y_train": y_train,
            "y_test": y_test,
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred
        })
        
        trained_models[name] = model
        print(f"✓ ({runtime:.2f}s)")
        
    except Exception as e:
        print(f"✗ Failed: {e}")

print(f"\n✓ Successfully trained {len(trained_models)} models")

rows = []

for res in model_results:
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(res['y_train'], res['y_train_pred']))
    test_rmse = np.sqrt(mean_squared_error(res['y_test'], res['y_test_pred']))
    
    train_mae = mean_absolute_error(res['y_train'], res['y_train_pred'])
    test_mae = mean_absolute_error(res['y_test'], res['y_test_pred'])
    
    train_r2 = r2_score(res['y_train'], res['y_train_pred'])
    test_r2 = r2_score(res['y_test'], res['y_test_pred'])
    
    try:
        train_mape = mean_absolute_percentage_error(res['y_train'], res['y_train_pred']) * 100
        test_mape = mean_absolute_percentage_error(res['y_test'], res['y_test_pred']) * 100
    except:
        train_mape = np.nan
        test_mape = np.nan
    
    diff_r2 = train_r2 - test_r2
    is_overfitting = diff_r2 > 0.15
    
    rows.append({
        "model_name": res["model_name"],
        "runtime": round(res["runtime"], 3),
        "train_rmse": round(train_rmse, 4),
        "test_rmse": round(test_rmse, 4),
        "train_mae": round(train_mae, 4),
        "test_mae": round(test_mae, 4),
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "train_mape": round(train_mape, 4),
        "test_mape": round(test_mape, 4),
        "r2_diff": round(diff_r2, 4),
        "is_overfitting": is_overfitting
    })

results_df = pd.DataFrame(rows)

# Mark best model
best_idx = results_df["test_r2"].idxmax()
results_df["is_best_model"] = False
results_df.loc[best_idx, "is_best_model"] = True


best_model_name = results_df.loc[best_idx, "model_name"]
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R²: {results_df.loc[best_idx, 'test_r2']:.4f}")
print(f"   Test RMSE: {results_df.loc[best_idx, 'test_rmse']:.4f}")
print(f"   Test MAE: {results_df.loc[best_idx, 'test_mae']:.4f}")


# Save the best model
print("\n" + "="*120)
print("SAVING BEST MODEL")
print("="*120)

# Create models directory if it doesn't exist
import os
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/sport_footware_model.pkl"

# Get the best model
#best_model = models[best_model_name]

best_model = results_df.loc[best_idx, "model_name"]

# Save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Model saved to: {MODEL_PATH}")