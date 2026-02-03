# Sport Footware Price Prediction

## 🎯 Problem Description

### Business Context
The global sports footwear market is highly competitive, with revenue optimization being crucial for success. This project addresses the challenge of **predicting final product revenues** for sports footwear based on various product features, market conditions, and temporal factors.

### Solution
We've developed an end-to-end machine learning solution that:
- **Predicts `revenue_usd`** for sports footwear products
- Analyzes multiple features including product attributes, regions, retailers, and temporal patterns
- Provides both batch and real-time predictions via a REST API
- Achieves **high accuracy** (R² > 0.95) using ensemble methods

### Use Cases
1. **Pricing Strategy**: Help retailers optimize pricing decisions
2. **Revenue Forecasting**: Predict future revenue based on product catalog
3. **Inventory Planning**: Estimate product values for inventory management
4. **Market Analysis**: Understand factors driving footwear prices

### Dataset
- **Source**: Global Sports Footwear Sales (2018-2026)
- **Size**: Multiple features across temporal, categorical, and numerical dimensions
- **Target Variable**: `final_price_usd` - the final selling price of footwear products
- **Features**: Product details, regional data, retailer information, date features, and more


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

## Hyperparameter Tuning

### Models with Parameter Grids:

**Ridge:**
- alpha: [0.1, 1.0, 10.0, 100.0, 1000.0]

**Random Forest:**
- n_estimators: [100, 200, 300]
- max_depth: [10, 20, 30]
- min_samples_split: [2, 5, 10]

**Gradient Boosting:**
- n_estimators: [100, 200, 300]
- learning_rate: [0.01, 0.1, 0.2]
- max_depth: [3, 5, 7]

**XGBoost:**
- n_estimators: [100, 200, 300]
- learning_rate: [0.01, 0.1, 0.2]
- max_depth: [3, 5, 7]

**LightGBM:**
- n_estimators: [100, 200, 300]
- learning_rate: [0.01, 0.1, 0.2]
- num_leaves: [31, 50, 70]

**CatBoost:**
- iterations: [100, 200, 300]
- learning_rate: [0.01, 0.1, 0.2]
- depth: [4, 6, 8]

### Tuning Strategy:
- **GridSearchCV** with 5-fold cross-validation
- **Scoring metric:** R² score
- **Parallel processing:** n_jobs=-1


## SHAP Analysis

### What is SHAP?
SHAP (SHapley Additive exPlanations) provides model-agnostic feature importance scores based on game theory.

### Visualizations Included:

1. **Summary Plot (Bar)** - Overall feature importance ranking
2. **Summary Plot (Beeswarm)** - Feature impact distribution
3. **Feature Importance Bar Chart** - Top 20 features
4. **Cumulative Importance** - How many features explain 80%/90% of predictions

### Interpretation:
- Higher SHAP value = greater impact on predictions
- Color indicates feature value (red = high, blue = low)
- Position shows impact direction (positive/negative)