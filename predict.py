# import pickle
# import pandas as pd
# from flask import Flask, request, jsonify

# # -----------------------------
# # Load trained model
# # -----------------------------
# MODEL_PATH = "models/sport_footware_model.pkl"

# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)


# # data_1 = {
# #         "age": 41,
# #         "incident_severity": "Total Loss",
# #         "total_claim_amount": 71600,
# #         "insured_hobbies": "chess",
# #         "policy_state": "OH",
# #         "number_of_vehicles_involved": 1,
# #         "property_damage": "YES",
# #         "auto_model": "92x",
# #         "insured_occupation": "craft-repair",
# #         "vehicle_claim": 52080,
# #         "bodily_injuries": 1,
# #         "months_as_customer": 328,
# #         "insured_relationship": "husband",
# #         "injury_claim": 6510,
# #         "insured_zip": 466132,
# #         "witnesses": 2,
# #         "capital-loss": 0,
# #         "authorities_contacted": "Police",
# #         "property_claim": 13020,
# #         "capital-gains": 53300,
# #         "incident_type": "Single Vehicle Collision",
# #         "insured_education_level": "MD",
# #         "collision_type": "Side Collision",
# #         "umbrella_limit": 0,
# #         "policy_number": 521585,
# #         "policy_csl": "250/500",
# #         "insured_sex": "MALE",
# #         "auto_year": 2004,
# #         "auto_make": "Saab",
# #         "policy_annual_premium": 1406.91,
# #         "police_report_available": "YES",
# #         "incident_hour_of_the_day": 5,
# #         "policy_deductable": 1000
# # }


# # -----------------------------
# # Initialize Flask app
# # -----------------------------
# app = Flask(__name__)


# @app.route("/", methods=["GET"])
# def health_check():
#     return jsonify({
#         "status": "ok",
#         "message": "Sportware Revenue Prediction API is running"
#     })

# @app.route("/predict", methods=["POST"])
# def predict():
#     data_1 = request.get_json()

#     X = pd.DataFrame([data_1])
#     fraud_prob  = model.predict_proba(X)[0,1]
#     fraud = fraud_prob >= 0.5

#     results = {
#             'Fraud Probability': float(fraud_prob),
#             'Fraud': bool(fraud)
#     }
#     return jsonify(results)


# # -----------------------------
# # Run server
# # -----------------------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

"""
Sport Footwear Revenue Prediction - Prediction Service
Flask API for serving predictions
"""

import pickle
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and metadata
model = None
label_encoders = None
feature_names = None
metadata = None


def load_model_artifacts():
    """Load model and related artifacts"""
    global model, label_encoders, feature_names, metadata
    
    print("Loading model artifacts...")
    
    # Load trained model
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded")
    
    # Load label encoders
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    print("✓ Label encoders loaded")
    
    # Load feature names
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("✓ Feature names loaded")
    
    # Load metadata
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    print("✓ Metadata loaded")
    
    print(f"\nModel: {metadata['model_name']}")
    print(f"Training date: {metadata['training_date']}")
    print(f"Test R²: {metadata['best_test_r2']:.4f}")
    print(f"Features: {len(feature_names)}")
    

def preprocess_input(data):
    """
    Preprocess input data to match training format
    
    Args:
        data: dict or list of dicts with input features
    
    Returns:
        DataFrame ready for prediction
    """
    # Convert to DataFrame if single dict
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    # Extract date features if order_date is provided
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['day_of_week'] = df['order_date'].dt.dayofweek
        df['quarter'] = df['order_date'].dt.quarter
        df = df.drop('order_date', axis=1)
    
    # Remove order_id if present
    if 'order_id' in df.columns:
        df = df.drop('order_id', axis=1)
    
    # Remove features that contain target information
    leaked_features = [
        'final_price_usd',  # Target itself
        'revenue_usd',      # This is final_price * units_sold - LEAKAGE!
        'discount_percent'
    ]

    df = df.drop(leaked_features, axis=1)
        
    # Encode categorical variables
    for col in label_encoders.keys():
        if col in df.columns:
            le = label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ 
                else -1
            )
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0  # Fill missing features with 0
    
    # Select only the features used in training (in correct order)
    df = df[feature_names]
    
    return df


@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'Sport Footwear Revenue Prediction API',
        'version': '1.0',
        'model': metadata['model_name'] if metadata else 'Unknown',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Make predictions (POST)',
            '/model_info': 'Model metadata'
        },
        'status': 'running'
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model_info')
def model_info():
    """Return model metadata"""
    if metadata is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_name': metadata['model_name'],
        'training_date': metadata['training_date'],
        'metrics': {
            'test_r2': metadata['best_test_r2'],
            'test_rmse': metadata['best_test_rmse'],
            'test_mae': metadata['best_test_mae']
        },
        'train_samples': metadata['train_samples'],
        'test_samples': metadata['test_samples'],
        'num_features': len(feature_names),
        'features': feature_names[:10]  # Show first 10 features
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions
    
    Expected JSON format:
    {
        "data": [
            {
                "feature1": value1,
                "feature2": value2,
                ...
            }
        ]
    }
    
    Or for single prediction:
    {
        "data": {
            "feature1": value1,
            "feature2": value2,
            ...
        }
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        json_data = request.get_json()
        
        if not json_data or 'data' not in json_data:
            return jsonify({
                'error': 'Invalid request format',
                'expected': {
                    'data': 'dict or list of dicts with features'
                }
            }), 400
        
        input_data = json_data['data']
        
        # Preprocess input
        X = preprocess_input(input_data)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Format response
        if isinstance(input_data, dict):
            # Single prediction
            response = {
                'prediction': float(predictions[0]),
                'predicted_price_usd': f"${predictions[0]:.2f}"
            }
        else:
            # Multiple predictions
            response = {
                'predictions': predictions.tolist(),
                'predicted_prices_usd': [f"${p:.2f}" for p in predictions],
                'count': len(predictions)
            }
        
        response['timestamp'] = datetime.now().isoformat()
        response['model'] = metadata['model_name']
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error making prediction'
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch predictions from CSV data
    
    Expected JSON format:
    {
        "data": [...] // list of feature dicts
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        json_data = request.get_json()
        
        if not json_data or 'data' not in json_data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        # Preprocess and predict
        X = preprocess_input(json_data['data'])
        predictions = model.predict(X)
        
        # Calculate statistics
        response = {
            'count': len(predictions),
            'predictions': predictions.tolist(),
            'statistics': {
                'mean': float(np.mean(predictions)),
                'median': float(np.median(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            },
            'timestamp': datetime.now().isoformat(),
            'model': metadata['model_name']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error making batch predictions'
        }), 500


def main():
    """Main function to run the Flask app"""
    # Load model artifacts
    load_model_artifacts()
    
    print("\n" + "="*60)
    print("SPORT FOOTWEAR PREDICTION API")
    print("="*60)
    print("\nStarting Flask server...")
    print("API will be available at: http://0.0.0.0:5000")
    print("\nAvailable endpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - Health check")
    print("  GET  /model_info - Model metadata")
    print("  POST /predict   - Make predictions")
    print("  POST /predict_batch - Batch predictions")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
