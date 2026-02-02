# """
# train.py
# --------
# Train final best model and save it to disk
# """

# # Import libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# import time
# from datetime import datetime

# # Machine Learning
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import (
#     mean_squared_error, 
#     mean_absolute_error, 
#     r2_score,
#     mean_absolute_percentage_error
# )

# # Models
# from sklearn.linear_model import Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import (
#     RandomForestRegressor, 
#     GradientBoostingRegressor,
#     ExtraTreesRegressor
# )
# from sklearn.neighbors import KNeighborsRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

# import shap
# import joblib

# # Set seed
# np.random.seed(42)

# # Plotting
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_palette('husl')

# print("All libraries imported successfully")

# # =========================
# # 1. Load data
# # =========================
# MODEL_PATH = "models/footware_model.pkl"

# df = pd.read_csv('./data/global_sports_footwear_sales_2018_2026.csv')

# # =========================
# # 2. Preprocessing
# # =========================

# # Make a copy
# data = df.copy()

# # # Remove rows with missing target
# # if data['final_price_usd'].isnull().sum() > 0:
# #     print(f"Removing {data['final_price_usd'].isnull().sum()} rows with missing target")
# #     data = data[~data['final_price_usd'].isnull()]

# # Separate target and features
# y = data['revenue_usd']

# # Remove features that contain target information
# leaked_features = [
#     'final_price_usd',  # Target itself
#     'revenue_usd',      # This is final_price * units_sold - LEAKAGE!
#     'order_id',         # Non-informative
#     'discount_percent'
# ]

# # Only keep legitimate features
# X = data.drop(columns=leaked_features)

# # Extract date features
# if 'order_date' in X.columns:
#     print("\nExtracting date features...")
#     X['order_date'] = pd.to_datetime(X['order_date'])
#     X['order_year'] = X['order_date'].dt.year
#     X['order_month'] = X['order_date'].dt.month
#     X['order_day'] = X['order_date'].dt.day
#     X['order_dayofweek'] = X['order_date'].dt.dayofweek
#     X['order_quarter'] = X['order_date'].dt.quarter
#     X = X.drop(columns=['order_date'])

# # Identify column types
# categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
# numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
# print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# # Encode categorical variables
# label_encoders = {}
# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = X[col].fillna('missing')
#     X[col] = le.fit_transform(X[col].astype(str))
#     label_encoders[col] = le
#     print(f"  Encoded '{col}': {len(le.classes_)} categories")

# # Fill missing numerical values
# for col in numerical_cols:
#     if X[col].isnull().sum() > 0:
#         median_val = X[col].median()
#         X[col] = X[col].fillna(median_val)
#         print(f"  Filled '{col}' with median: {median_val}")

# print(f"\n✓ Preprocessing complete")
# print(f"  Final shape: {X.shape}")
# print(f"  No missing values: {X.isnull().sum().sum() == 0}")


# # =========================
# # 3. Train-test split
# # =========================

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print(f"Train set: {X_train.shape[0]} samples")
# print(f"Test set: {X_test.shape[0]} samples")
# print(f"Features: {X_train.shape[1]}")

# # =========================
# # 4. Training Model
# # =========================

# # Define regressors
# regressors = [
#     ['Ridge', Ridge(random_state=42)],
#     ['Lasso', Lasso(random_state=42, max_iter=5000)],
#     ['ElasticNet', ElasticNet(random_state=42, max_iter=5000)],
#     ['GradientBoosting', GradientBoostingRegressor(random_state=42)],
#     ['XGBoost', XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)],
#     ['LightGBM', LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)],
#     ['CatBoost', CatBoostRegressor(random_state=42, verbose=0)]
# ]

# model_results = []
# trained_models = {}

# print("Training models...\n")

# for name, model in regressors:
#     print(f"Training {name}...", end=' ')
#     start_time = time.time()
    
#     try:
#         # Train
#         model.fit(X_train, y_train)
#         runtime = time.time() - start_time
        
#         # Predictions
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)
        
#         # Store
#         model_results.append({
#             "model_name": name,
#             "model": model,
#             "runtime": runtime,
#             "y_train": y_train,
#             "y_test": y_test,
#             "y_train_pred": y_train_pred,
#             "y_test_pred": y_test_pred
#         })
        
#         trained_models[name] = model
#         print(f"✓ ({runtime:.2f}s)")
        
#     except Exception as e:
#         print(f"✗ Failed: {e}")

# print(f"\n✓ Successfully trained {len(trained_models)} models")

# rows = []

# for res in model_results:
#     # Metrics
#     train_rmse = np.sqrt(mean_squared_error(res['y_train'], res['y_train_pred']))
#     test_rmse = np.sqrt(mean_squared_error(res['y_test'], res['y_test_pred']))
    
#     train_mae = mean_absolute_error(res['y_train'], res['y_train_pred'])
#     test_mae = mean_absolute_error(res['y_test'], res['y_test_pred'])
    
#     train_r2 = r2_score(res['y_train'], res['y_train_pred'])
#     test_r2 = r2_score(res['y_test'], res['y_test_pred'])
    
#     try:
#         train_mape = mean_absolute_percentage_error(res['y_train'], res['y_train_pred']) * 100
#         test_mape = mean_absolute_percentage_error(res['y_test'], res['y_test_pred']) * 100
#     except:
#         train_mape = np.nan
#         test_mape = np.nan
    
#     diff_r2 = train_r2 - test_r2
#     is_overfitting = diff_r2 > 0.15
    
#     rows.append({
#         "model_name": res["model_name"],
#         "runtime": round(res["runtime"], 3),
#         "train_rmse": round(train_rmse, 4),
#         "test_rmse": round(test_rmse, 4),
#         "train_mae": round(train_mae, 4),
#         "test_mae": round(test_mae, 4),
#         "train_r2": round(train_r2, 4),
#         "test_r2": round(test_r2, 4),
#         "train_mape": round(train_mape, 4),
#         "test_mape": round(test_mape, 4),
#         "r2_diff": round(diff_r2, 4),
#         "is_overfitting": is_overfitting
#     })

# results_df = pd.DataFrame(rows)

# # Mark best model
# best_idx = results_df["test_r2"].idxmax()
# results_df["is_best_model"] = False
# results_df.loc[best_idx, "is_best_model"] = True


# best_model_name = results_df.loc[best_idx, "model_name"]
# print(f"\n🏆 BEST MODEL: {best_model_name}")
# print(f"   Test R²: {results_df.loc[best_idx, 'test_r2']:.4f}")
# print(f"   Test RMSE: {results_df.loc[best_idx, 'test_rmse']:.4f}")
# print(f"   Test MAE: {results_df.loc[best_idx, 'test_mae']:.4f}")


# # Save the best model
# print("\n" + "="*120)
# print("SAVING BEST MODEL")
# print("="*120)

# # Create models directory if it doesn't exist
# import os
# os.makedirs("models", exist_ok=True)

# MODEL_PATH = "models/sport_footware_model.pkl"

# # Get the best model
# #best_model = models[best_model_name]

# best_model = results_df.loc[best_idx, "model_name"]

# # Save model
# with open(MODEL_PATH, 'wb') as f:
#     pickle.dump(best_model, f)
# print(f"✓ Model saved to: {MODEL_PATH}")

# # Save label encoders
# ENCODERS_PATH = "models/label_encoders.pkl"
# with open(ENCODERS_PATH, 'wb') as f:
#     pickle.dump(label_encoders, f)
# print(f"✓ Label encoders saved to: {ENCODERS_PATH}")

# # Save feature names
# FEATURES_PATH = "models/feature_names.pkl"
# with open(FEATURES_PATH, 'wb') as f:
#     pickle.dump(feature_names, f)
# print(f"✓ Feature names saved to: {FEATURES_PATH}")

# # Save metadata
# METADATA_PATH = "models/model_metadata.json"
# metadata = {
#     'model_name': best_model_name,
#     'model_type': type(best_model).__name__,
#     'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#     'feature_names': feature_names,
#     'num_features': len(feature_names),
#     'metrics': {
#         'train_r2': float(results_df.loc[best_idx, 'train_r2']),
#         'test_r2': float(results_df.loc[best_idx, 'test_r2']),
#         'train_rmse': float(results_df.loc[best_idx, 'train_rmse']),
#         'test_rmse': float(results_df.loc[best_idx, 'test_rmse']),
#         'train_mae': float(results_df.loc[best_idx, 'train_mae']),
#         'test_mae': float(results_df.loc[best_idx, 'test_mae']),
#         'overfitting_gap': float(results_df.loc[best_idx, 'train_r2'] - results_df.loc[best_idx, 'test_r2'])
#     },
#     'training_samples': len(X_train),
#     'test_samples': len(X_test),
#     'categorical_features': list(label_encoders.keys())
# }

# with open(METADATA_PATH, 'w') as f:
#     json.dump(metadata, f, indent=4)
# print(f"✓ Metadata saved to: {METADATA_PATH}")

# print("\n" + "="*120)
# print("MODEL ARTIFACTS SAVED SUCCESSFULLY!")
# print("="*120)
# print("\nSaved files:")
# print(f"  1. {MODEL_PATH} - Trained model")
# print(f"  2. {ENCODERS_PATH} - Label encoders for categorical features")
# print(f"  3. {FEATURES_PATH} - List of feature names")
# print(f"  4. {METADATA_PATH} - Model metadata and metrics")


"""
Sport Footwear Revenue Prediction - Training Script
This script trains multiple regression models to predict final_price_usd
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# SHAP for feature importance
import shap
import matplotlib.pyplot as plt

class SportFootwearModel:
    """Class to handle model training and evaluation"""
    
    def __init__(self, data_path='./data/global_sports_footwear_sales_2018_2026.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load and initial preprocessing of data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        
        # Remove rows with missing target variable
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['revenue_usd'])
        print(f"Removed {initial_rows - len(self.df)} rows with missing target")
        
        # Handle missing values
        # Numerical features - fill with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"Filled {col} with median: {median_val}")
        
        # Categorical features - fill with 'missing'
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna('missing', inplace=True)
                print(f"Filled {col} with 'missing'")
        
        # Feature Engineering - Date features
        if 'order_date' in self.df.columns:
            print("\nExtracting date features...")
            self.df['order_date'] = pd.to_datetime(self.df['order_date'])
            self.df['year'] = self.df['order_date'].dt.year
            self.df['month'] = self.df['order_date'].dt.month
            self.df['day'] = self.df['order_date'].dt.day
            self.df['day_of_week'] = self.df['order_date'].dt.dayofweek
            self.df['quarter'] = self.df['order_date'].dt.quarter
            print("Date features extracted: year, month, day, day_of_week, quarter")
        
        # Remove non-informative columns
        cols_to_drop = ['order_id', 'order_date']
        existing_cols_to_drop = [col for col in cols_to_drop if col in self.df.columns]
        if existing_cols_to_drop:
            self.df = self.df.drop(columns=existing_cols_to_drop)
            print(f"\nDropped columns: {existing_cols_to_drop}")
        
        # Encode categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print(f"\nEncoding categorical features: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  {col}: {len(le.classes_)} unique values")
        
        print(f"\nFinal dataset shape: {self.df.shape}")
        return self.df
    
    def prepare_features(self, test_size=0.2, random_state=42):
        """Prepare features and target for training"""
        print("\n" + "="*60)
        print("PREPARING FEATURES")
        print("="*60)
        
        # Separate features and target

        # Remove features that contain target information
        leaked_features = [
            'final_price_usd',  # Target itself
            'revenue_usd',      # This is final_price * units_sold - LEAKAGE!
            'discount_percent'
        ]

        X = self.df.drop(leaked_features, axis=1)
        y = self.df['revenue_usd']
        
        self.feature_names = X.columns.tolist()
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Target variable: revenue_usd")
        print(f"  Mean: ${y.mean():.2f}")
        print(f"  Std: ${y.std():.2f}")
        print(f"  Min: ${y.min():.2f}")
        print(f"  Max: ${y.max():.2f}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def train_models(self):
        """Train all baseline models"""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        # Define models
        models_dict = {
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbosity=-1),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
        }
        
        # Train each model
        for name, model in models_dict.items():
            print(f"\nTraining {name}...")
            start_time = datetime.now()
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model
            self.models[name] = model
            
            # Store results
            result = {
                'model': name,
                'train_r2': train_metrics['r2'],
                'test_r2': test_metrics['r2'],
                'train_rmse': train_metrics['rmse'],
                'test_rmse': test_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'test_mae': test_metrics['mae'],
                'train_mape': train_metrics['mape'],
                'test_mape': test_metrics['mape'],
                'overfitting': train_metrics['r2'] - test_metrics['r2'],
                'training_time': training_time
            }
            self.results.append(result)
            
            # Print results
            print(f"  Train R²: {train_metrics['r2']:.4f}")
            print(f"  Test R²: {test_metrics['r2']:.4f}")
            print(f"  Test RMSE: ${test_metrics['rmse']:.2f}")
            print(f"  Test MAE: ${test_metrics['mae']:.2f}")
            print(f"  Test MAPE: {test_metrics['mape']:.2f}%")
            print(f"  Overfitting: {result['overfitting']:.4f}")
            print(f"  Training Time: {training_time:.2f}s")
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(self.results)
        self.results_df = self.results_df.sort_values('test_r2', ascending=False)
        
        # Identify best model
        self.best_model_name = self.results_df.iloc[0]['model']
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(self.results_df.to_string(index=False))
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Test R²: {self.results_df.iloc[0]['test_r2']:.4f}")
        
        return self.results_df
    
    def tune_best_model(self):
        """Hyperparameter tuning for the best model"""
        print("\n" + "="*60)
        print(f"HYPERPARAMETER TUNING - {self.best_model_name}")
        print("="*60)
        
        # Define parameter grids for different models
        param_grids = {
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 70]
            },
            'CatBoost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [4, 6, 8]
            }
        }
        
        if self.best_model_name not in param_grids:
            print(f"No parameter grid defined for {self.best_model_name}")
            print("Using default best model without tuning.")
            return self.best_model
        
        print(f"Parameter grid: {param_grids[self.best_model_name]}")
        
        # Create base model
        if self.best_model_name == 'XGBoost':
            base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
        elif self.best_model_name == 'LightGBM':
            base_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
        elif self.best_model_name == 'CatBoost':
            base_model = CatBoostRegressor(random_state=42, verbose=0)
        elif self.best_model_name == 'GradientBoosting':
            base_model = GradientBoostingRegressor(random_state=42)
        elif self.best_model_name == 'Ridge':
            base_model = Ridge(random_state=42)
        else:
            base_model = self.best_model
        
        # Grid search
        print("\nPerforming GridSearchCV...")
        grid_search = GridSearchCV(
            base_model,
            param_grids[self.best_model_name],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        self.models[f'{self.best_model_name}_tuned'] = self.best_model
        
        # Evaluate tuned model
        y_train_pred = self.best_model.predict(self.X_train)
        y_test_pred = self.best_model.predict(self.X_test)
        
        train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
        test_metrics = self.calculate_metrics(self.y_test, y_test_pred)
        
        print("\n" + "="*60)
        print("TUNING RESULTS")
        print("="*60)
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"\nTuned Model Performance:")
        print(f"  Train R²: {train_metrics['r2']:.4f}")
        print(f"  Test R²: {test_metrics['r2']:.4f}")
        print(f"  Test RMSE: ${test_metrics['rmse']:.2f}")
        print(f"  Test MAE: ${test_metrics['mae']:.2f}")
        print(f"  Test MAPE: {test_metrics['mape']:.2f}%")
        print(f"  Overfitting: {train_metrics['r2'] - test_metrics['r2']:.4f}")
        
        # Save tuned results
        tuned_result = {
            'model': f'{self.best_model_name}_tuned',
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'train_rmse': train_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_mape': train_metrics['mape'],
            'test_mape': test_metrics['mape'],
            'overfitting': train_metrics['r2'] - test_metrics['r2'],
            'best_params': grid_search.best_params_
        }
        
        return self.best_model, tuned_result
    
    def analyze_feature_importance(self, save_plots=True):
        """Analyze feature importance using SHAP"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS (SHAP)")
        print("="*60)
        
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(self.best_model, self.X_train)
            shap_values = explainer(self.X_test)
            
            if save_plots:
                # Summary plot (bar)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False)
                plt.title("SHAP Feature Importance - Bar Plot")
                plt.tight_layout()
                plt.savefig('shap_importance_bar.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("Saved: shap_importance_bar.png")
                
                # Summary plot (beeswarm)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, self.X_test, show=False)
                plt.title("SHAP Feature Importance - Beeswarm Plot")
                plt.tight_layout()
                plt.savefig('shap_importance_beeswarm.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("Saved: shap_importance_beeswarm.png")
            
            # Calculate mean absolute SHAP values
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(shap_values.values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
            
            return feature_importance
            
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")
            print("Continuing without SHAP analysis...")
            return None
    
    def save_model(self, model_filename='model.pkl', metadata_filename='models/model_metadata.json'):
        """Save the trained model and metadata"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        # Save model
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Model saved: {model_filename}")
        
        # Save label encoders
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"Label encoders saved: label_encoders.pkl")
        
        # Save feature names
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"Feature names saved: feature_names.pkl")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'best_test_r2': float(self.results_df.iloc[0]['test_r2']),
            'best_test_rmse': float(self.results_df.iloc[0]['test_rmse']),
            'best_test_mae': float(self.results_df.iloc[0]['test_mae']),
        }
        
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved: {metadata_filename}")
        
        print("\nAll files saved successfully!")
        return model_filename, metadata_filename


def main():
    """Main training pipeline"""
    print("="*60)
    print("SPORT FOOTWEAR REVENUE PREDICTION - TRAINING PIPELINE")
    print("="*60)
    
    # Initialize model
    model = SportFootwearModel(data_path='./data/global_sports_footwear_sales_2018_2026.csv')
    
    # Load data
    model.load_data()
    
    # Preprocess data
    model.preprocess_data()
    
    # Prepare features
    model.prepare_features()
    
    # Train baseline models
    model.train_models()
    
    # Tune best model
    model.tune_best_model()
    
    # Analyze feature importance
    model.analyze_feature_importance()
    
    # Save model
    model.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModel files created:")
    print("  - model.pkl (trained model)")
    print("  - label_encoders.pkl (categorical encoders)")
    print("  - feature_names.pkl (feature list)")
    print("  - model_metadata.json (training metadata)")
    print("  - shap_importance_bar.png (SHAP bar plot)")
    print("  - shap_importance_beeswarm.png (SHAP beeswarm plot)")


if __name__ == "__main__":
    main()
