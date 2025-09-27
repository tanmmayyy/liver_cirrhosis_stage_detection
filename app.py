"""
Integrated Flask Application for Liver Cirrhosis Stage Detection
This file integrates with your existing project structure
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import json
import logging
from datetime import datetime
import joblib

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiverCirrhosisPredictor:
    """Main prediction class using your existing model structure"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_loaded = False
        
        # Try to load existing model
        self.load_model()
    
    def load_model(self):
        """Load the trained model from your model directory"""
        model_path = 'model/liver_cirrhosis_model.pkl'
        
        try:
            if os.path.exists(model_path):
                # Load the model using joblib
                model_data = joblib.load(model_path)
                
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.feature_names = model_data.get('feature_names', [])
                else:
                    # If it's just the model object
                    self.model = model_data
                
                self.is_loaded = True
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning(f"Model file not found at {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def create_advanced_features(self, df):
        """Create the same advanced features as in your feature engineering"""
        df_features = df.copy()
        
        # Age conversion
        if 'Age' in df_features.columns:
            df_features['Age_Years'] = df_features['Age'] / 365.25
        
        # Clinical ratios
        df_features['Bilirubin_Albumin_Ratio'] = df_features['Bilirubin'] / (df_features['Albumin'] + 0.001)
        df_features['Copper_Albumin_Ratio'] = df_features['Copper'] / (df_features['Albumin'] + 0.001)
        df_features['SGOT_Platelets_Ratio'] = df_features['SGOT'] / (df_features['Platelets'] + 0.001)
        df_features['AlkPhos_Albumin_Ratio'] = df_features['Alk_Phos'] / (df_features['Albumin'] + 0.001)
        df_features['Prothrombin_Albumin_Ratio'] = df_features['Prothrombin'] / (df_features['Albumin'] + 0.001)
        
        # Composite scores
        bilirubin_norm = np.clip((df_features['Bilirubin'] - 0.3) / 19.7, 0, 1)
        albumin_norm = np.clip(1 - (df_features['Albumin'] - 2.0) / 3.0, 0, 1)
        prothrombin_norm = np.clip((df_features['Prothrombin'] - 10) / 15, 0, 1)
        
        df_features['Liver_Function_Score'] = (bilirubin_norm + albumin_norm + prothrombin_norm) / 3
        
        # Clinical complications
        ascites_score = (df_features['Ascites'] == 'Y').astype(int)
        hepato_score = (df_features['Hepatomegaly'] == 'Y').astype(int)
        spiders_score = (df_features['Spiders'] == 'Y').astype(int)
        edema_score = df_features['Edema'].map({'N': 0, 'S': 1, 'Y': 2}).fillna(0)
        
        df_features['Portal_Hypertension_Score'] = ascites_score + hepato_score + spiders_score
        df_features['Fluid_Retention_Score'] = ascites_score + (edema_score > 0).astype(int)
        
        # Severity indicators
        df_features['Bilirubin_Severe'] = (df_features['Bilirubin'] > 3.0).astype(int)
        df_features['Albumin_Low'] = (df_features['Albumin'] < 3.0).astype(int)
        df_features['Platelets_Low'] = (df_features['Platelets'] < 150).astype(int)
        df_features['Prothrombin_High'] = (df_features['Prothrombin'] > 13).astype(int)
        
        # Log transformations
        df_features['Bilirubin_Log'] = np.log1p(df_features['Bilirubin'])
        df_features['Copper_Log'] = np.log1p(df_features['Copper'])
        df_features['Alk_Phos_Log'] = np.log1p(df_features['Alk_Phos'])
        
        # Polynomial features for key biomarkers
        df_features['Bilirubin_Squared'] = df_features['Bilirubin'] ** 2
        df_features['Albumin_Squared'] = df_features['Albumin'] ** 2
        
        # Interaction features
        df_features['Bilirubin_x_Prothrombin'] = df_features['Bilirubin'] * df_features['Prothrombin']
        df_features['Age_x_Bilirubin'] = df_features['Age_Years'] * df_features['Bilirubin']
        
        return df_features
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        # Binary encoding
        binary_features = ['Ascites', 'Hepatomegaly', 'Spiders']
        for feature in binary_features:
            if feature in df_encoded.columns:
                df_encoded[f'{feature}_Binary'] = (df_encoded[feature] == 'Y').astype(int)
        
        # Ordinal encoding for Edema
        edema_mapping = {'N': 0, 'S': 1, 'Y': 2}
        if 'Edema' in df_encoded.columns:
            df_encoded['Edema_Ordinal'] = df_encoded['Edema'].map(edema_mapping).fillna(0)
        
        # One-hot encoding for nominal features
        if 'Sex' in df_encoded.columns:
            df_encoded['Sex_M'] = (df_encoded['Sex'] == 'M').astype(int)
            df_encoded['Sex_F'] = (df_encoded['Sex'] == 'F').astype(int)
        
        if 'Drug' in df_encoded.columns:
            df_encoded['Drug_D-penicillamine'] = (df_encoded['Drug'] == 'D-penicillamine').astype(int)
            df_encoded['Drug_Placebo'] = (df_encoded['Drug'] == 'Placebo').astype(int)
        
        return df_encoded
    
    def prepare_prediction_data(self, patient_data):
        """Prepare patient data for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Create advanced features
        df_features = self.create_advanced_features(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)
        
        # Select and order features for prediction
        prediction_features = [
            'Age_Years', 'N_Days', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
            'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin',
            'Bilirubin_Albumin_Ratio', 'Copper_Albumin_Ratio', 'SGOT_Platelets_Ratio',
            'AlkPhos_Albumin_Ratio', 'Prothrombin_Albumin_Ratio', 'Liver_Function_Score',
            'Portal_Hypertension_Score', 'Fluid_Retention_Score', 'Bilirubin_Severe',
            'Albumin_Low', 'Platelets_Low', 'Prothrombin_High', 'Bilirubin_Log',
            'Copper_Log', 'Alk_Phos_Log', 'Bilirubin_Squared', 'Albumin_Squared',
            'Bilirubin_x_Prothrombin', 'Age_x_Bilirubin', 'Ascites_Binary',
            'Hepatomegaly_Binary', 'Spiders_Binary', 'Edema_Ordinal',
            'Sex_M', 'Sex_F', 'Drug_D-penicillamine', 'Drug_Placebo'
        ]
        
        # Ensure all features exist
        for feature in prediction_features:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Select final feature set
        X = df_encoded[prediction_features]
        
        return X
    
    def predict(self, patient_data):
        """Make prediction for patient data"""
        try:
            if not self.is_loaded:
                return None
            
            # Prepare data
            X = self.prepare_prediction_data(patient_data)
            
            # Scale if scaler is available
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Make prediction
            stage_pred = self.model.predict(X_scaled)[0]
            stage_proba = self.model.predict_proba(X_scaled)[0]
            
            # Calculate feature importance for this prediction
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, self.model.feature_importances_))
                # Get top 5 most important features
                top_features = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True)[:5])
            else:
                top_features = {}
            
            return {
                'stage': int(stage_pred),
                'confidence': float(stage_proba[stage_pred - 1]),
                'probabilities': {
                    'stage_1': float(stage_proba[0]),
                    'stage_2': float(stage_proba[1]),
                    'stage_3': float(stage_proba[2])
                },
                'feature_importance': top_features
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

# Initialize predictor
predictor = LiverCirrhosisPredictor()

@app.route('/')
def index():
    """Serve the main application"""
    return render_template('index.html')

@app.route('/app')
def app_interface():
    """Alternative route for the app"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_stage():
    """API endpoint for prediction"""
    try:
        # Get JSON data
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'Age', 'Sex', 'Drug', 'N_Days', 'Ascites', 'Hepatomegaly', 
            'Spiders', 'Edema', 'Bilirubin', 'Cholesterol', 'Albumin', 
            'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin'
        ]
        
        # Check for missing fields (allow frontend field names too)
        frontend_to_backend = {
            'age': 'Age', 'sex': 'Sex', 'drug': 'Drug', 'n_days': 'N_Days',
            'ascites': 'Ascites', 'hepatomegaly': 'Hepatomegaly', 'spiders': 'Spiders',
            'edema': 'Edema', 'bilirubin': 'Bilirubin', 'cholesterol': 'Cholesterol',
            'albumin': 'Albumin', 'copper': 'Copper', 'alk_phos': 'Alk_Phos',
            'sgot': 'SGOT', 'tryglicerides': 'Tryglicerides', 'platelets': 'Platelets',
            'prothrombin': 'Prothrombin'
        }
        
        # Convert frontend field names to backend field names
        converted_data = {}
        for frontend_key, backend_key in frontend_to_backend.items():
            if frontend_key in patient_data:
                if frontend_key == 'age':
                    # Convert age to days if it's in years
                    age_value = float(patient_data[frontend_key])
                    if age_value < 200:  # Assume it's in years
                        converted_data[backend_key] = age_value * 365.25
                    else:
                        converted_data[backend_key] = age_value
                else:
                    converted_data[backend_key] = patient_data[frontend_key]
            elif backend_key in patient_data:
                converted_data[backend_key] = patient_data[backend_key]
        
        # Make prediction
        prediction = predictor.predict(converted_data)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    return jsonify({
        'model_loaded': predictor.is_loaded,
        'model_type': type(predictor.model).__name__ if predictor.model else None,
        'features_count': len(predictor.feature_names),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_ready': predictor.is_loaded,
        'timestamp': datetime.now().isoformat()
    })

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('app/static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('app/templates', exist_ok=True)
    os.makedirs('app/static', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Check if model exists
    if not predictor.is_loaded:
        print("\n" + "="*60)
        print("⚠️  WARNING: No trained model found!")
        print("Please run the training script first:")
        print("1. Run your EDA notebook: jupyter notebook notebook/eda.ipynb")
        print("2. Train your model using the feature engineering code")
        print("3. Save the model to: model/liver_cirrhosis_model.pkl")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("🎉 Liver Cirrhosis Detection System Ready!")
        print("🔗 Access the app at: http://localhost:5000")
        print("📊 API Documentation: http://localhost:5000/api/health")
        print("="*60 + "\n")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)