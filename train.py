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