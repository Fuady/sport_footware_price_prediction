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


## 📁 Project Structure

```
sport_footware_price_prediction/
│
├── data/
│   └── Global_Sports_Footwear_Sales.csv    # Dataset
│
├── notebook.ipynb                           # EDA and experimentation
├── train.py                                 # Training script
├── predict.py                               # Flask API for predictions
├── prediction_test.ipynb                    # Testing notebook
│
├── model.pkl                                # Trained model (generated)
├── label_encoders.pkl                       # Categorical encoders (generated)
├── feature_names.pkl                        # Feature list (generated)
├── model_metadata.json                      # Model metadata (generated)
│
├── requirements.txt                         # Python dependencies
├── Dockerfile                               # Container configuration
├── README.md                                # Documentation
└── .gitignore                               # Git ignore rules
```

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

###  **Feature Analysis**

**Categorical Features:**
- `region`: Geographic distribution of sales
- `brand`: Brand-wise price variations
- `product_category`: Category-specific pricing patterns
- `retailer`: Retailer impact on pricing
- `distribution_channel`: Online vs offline pricing differences

**Numerical Features:**
- `units_sold`: Correlation with pricing
- `discount_percentage`: Impact on final price
- `rating`: Quality indicator
- Date features: Temporal patterns

###  **Feature Importance Analysis**
Conducted using multiple methods:
- **Correlation Analysis**: Heatmaps showing feature relationships
- **SHAP Values**: Model-agnostic feature importance
- **Tree-based Importance**: Feature importance from ensemble models

###  **Key Insights**
- Strong seasonal patterns in pricing
- Regional variations significantly impact prices
- Brand and product category are top price drivers
- Discount strategies vary by distribution channel


## Model Training

### Models Trained

We trained and compared **7 different regression models**:

| Model | Type | Description |
|-------|------|-------------|
| **Ridge** | Linear | L2 regularization for feature stability |
| **Lasso** | Linear | L1 regularization for feature selection |
| **ElasticNet** | Linear | Combined L1 + L2 regularization |
| **Gradient Boosting** | Tree-based | Sequential ensemble learning |
| **XGBoost** | Tree-based | Optimized gradient boosting |
| **LightGBM** | Tree-based | Fast gradient boosting framework |
| **CatBoost** | Tree-based | Gradient boosting for categorical features |


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


## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Option 1: Local Setup (Recommended)

#### Step 1: Clone the Repository
```bash
git clone https://github.com/Fuady/sport_footware_price_prediction.git
cd sport_footware_price_prediction
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Verify Installation
```bash
python -c "import pandas, sklearn, xgboost, lightgbm, catboost; print('All dependencies installed successfully!')"
```


## 🌐 Model Deployment

### Local Deployment

The model is deployed as a **Flask REST API** that can be run locally:

```bash
# Start the API server
python predict.py
```

The server will start on `http://0.0.0.0:5000`

### API Features
- ✅ RESTful design
- ✅ JSON request/response format
- ✅ Single and batch predictions
- ✅ Health check endpoint
- ✅ Model metadata endpoint
- ✅ Error handling
- ✅ Input validation

### Production Deployment Options

#### 1. **Using Gunicorn (Production WSGI Server)**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 predict:app
```

#### 2. **Using Docker** (See Containerization section)

#### 3. **Using Cloud Services** (See Cloud Deployment section)

---

## 🐳 Containerization

### Docker Setup

#### Building the Docker Image

```bash
# Build the image
docker build -t sport-footwear-predictor .

# Verify the image
docker images | grep sport-footwear-predictor
```

#### Running the Container

**Option 1: Run Prediction API**
```bash
docker run -p 5000:5000 sport-footwear-predictor
```

**Option 2: Train Model in Container**
```bash
docker run sport-footwear-predictor python train.py
```

**Option 3: Interactive Mode**
```bash
docker run -it sport-footwear-predictor /bin/bash
```

#### Docker Commands

```bash
# Run with volume mounting (for persistent data)
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  sport-footwear-predictor

# Run in detached mode
docker run -d -p 5000:5000 --name footwear-api sport-footwear-predictor

# View logs
docker logs footwear-api

# Stop container
docker stop footwear-api

# Remove container
docker rm footwear-api
```

#### Testing the Dockerized API

```bash
# Health check
curl http://localhost:5000/health

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"brand": "Nike", "region": "US", ...}}'
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```