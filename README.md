# Sport Footware Price Prediction

## Overview
Complete machine learning pipeline for predicting `final_price_usd` in the Global Sports Footwear Sales dataset (2018-2026).

## Objectives
- Train and compare **7 regression models**
- Perform **hyperparameter tuning** on the best model
- Generate **SHAP feature importance** analysis
- Provide comprehensive **visualizations and metrics**

## Models Included

1. **Ridge Regression** - Linear model with L2 regularization
2. **Lasso Regression** - Linear model with L1 regularization
3. **ElasticNet** - Linear model with L1 + L2 regularization
4. **Gradient Boosting** - Sequential tree ensemble
5. **XGBoost** - Optimized gradient boosting
6. **LightGBM** - Fast gradient boosting
7. **CatBoost** - Gradient boosting for categorical features

## Evaluation Metrics

### Regression Metrics
- **R² Score** - Coefficient of determination (higher is better)
- **RMSE** - Root Mean Squared Error (lower is better)
- **MAE** - Mean Absolute Error (lower is better)
- **MAPE** - Mean Absolute Percentage Error (lower is better)

### Additional Analysis
- **Overfitting Detection** - Train vs Test R² difference
- **Training Time** - Model efficiency comparison
- **Feature Importance** - SHAP values

## Data Preprocessing

### Steps Performed:
1. **Missing Value Handling**
   - Remove rows with missing target variable
   - Fill numerical features with median
   - Fill categorical features with 'missing' label

2. **Feature Engineering**
   - Extract date components (year, month, day, day of week, quarter)
   - Remove non-informative columns (order_id)

3. **Encoding**
   - Label encoding for categorical variables
   - Maintain original feature names for interpretability

4. **Train-Test Split**
   - 80% training, 20% testing
   - Random state = 42 for reproducibility