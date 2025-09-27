"""
Flask Backend for Liver Cirrhosis Stage Detection
Modern API with ML model integration, feature engineering, and comprehensive endpoints
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import os
import json
from werkzeug.exceptions import BadRequest
import traceback

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiverCirrhosisModel:
    """
    Main model class for liver cirrhosis stage prediction
    Includes feature engineering, model training, and prediction methods
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.feature_importance = {}
        self.model_trained = False
        self.model_accuracy = 0.0
        
    def create_features(self, df):
        """Create engineered features from raw patient data"""
        df_features = df.copy()
        
        # Convert age from days to years if needed
        if 'Age' in df_features.columns and df_features['Age'].max() > 200:
            df_features['Age_Years'] = df_features['Age'] / 365.25
        else:
            df_features['Age_Years'] = df_features.get('Age', df_features.get('age', 0))
        
        # Clinical ratios (key biomarkers)
        df_features['Bilirubin_Albumin_Ratio'] = df_features['Bilirubin'] / (df_features['Albumin'] + 0.001)
        df_features['Copper_Albumin_Ratio'] = df_features['Copper'] / (df_features['Albumin'] + 0.001)
        df_features['SGOT_Platelets_Ratio'] = df_features['SGOT'] / (df_features['Platelets'] + 0.001)
        df_features['Prothrombin_Albumin_Ratio'] = df_features['Prothrombin'] / (df_features['Albumin'] + 0.001)
        
        # Composite scores
        bilirubin_norm = (df_features['Bilirubin'] - 0.3) / (20 - 0.3)  # Normalize bilirubin
        albumin_norm = 1 - (df_features['Albumin'] - 2.0) / (5.0 - 2.0)  # Invert albumin (lower is worse)
        prothrombin_norm = (df_features['Prothrombin'] - 10) / (20 - 10)  # Normalize prothrombin
        
        df_features['Liver_Function_Score'] = np.clip(
            (bilirubin_norm + albumin_norm + prothrombin_norm) / 3, 0, 1
        )
        
        # Clinical complications score
        ascites_score = (df_features['Ascites'] == 'Y').astype(int)
        edema_score = (df_features['Edema'].isin(['S', 'Y'])).astype(int)
        hepato_score = (df_features['Hepatomegaly'] == 'Y').astype(int)
        spiders_score = (df_features['Spiders'] == 'Y').astype(int)
        
        df_features['Clinical_Complications_Score'] = ascites_score + edema_score + hepato_score + spiders_score
        
        # Severity categories
        df_features['Bilirubin_Severe'] = (df_features['Bilirubin'] > 3.0).astype(int)
        df_features['Albumin_Low'] = (df_features['Albumin'] < 3.0).astype(int)
        df_features['Platelets_Low'] = (df_features['Platelets'] < 150).astype(int)
        df_features['Prothrombin_High'] = (df_features['Prothrombin'] > 13).astype(int)
        
        # Log transformations for skewed features
        df_features['Bilirubin_Log'] = np.log1p(df_features['Bilirubin'])
        df_features['Copper_Log'] = np.log1p(df_features['Copper'])
        df_features['Alk_Phos_Log'] = np.log1p(df_features['Alk_Phos'])
        
        return df_features
    
    def encode_categorical_features(self, df):
        """Encode categorical features for model training"""
        df_encoded = df.copy()
        
        # Binary encoding for yes/no features
        binary_features = ['Ascites', 'Hepatomegaly', 'Spiders']
        for feature in binary_features:
            if feature in df_encoded.columns:
                df_encoded[f'{feature}_Binary'] = (df_encoded[feature] == 'Y').astype(int)
        
        # Ordinal encoding for Edema (N=0, S=1, Y=2)
        if 'Edema' in df_encoded.columns:
            edema_mapping = {'N': 0, 'S': 1, 'Y': 2}
            df_encoded['Edema_Ordinal'] = df_encoded['Edema'].map(edema_mapping).fillna(0)
        
        # One-hot encoding for nominal features
        nominal_features = ['Drug', 'Sex']
        for feature in nominal_features:
            if feature in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[feature], prefix=feature)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        return df_encoded
    
    def prepare_features(self, df):
        """Complete feature preparation pipeline"""
        # Create engineered features
        df_features = self.create_features(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)
        
        # Select final feature set for modeling
        feature_columns = [
            'Age_Years', 'N_Days', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
            'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin',
            'Bilirubin_Albumin_Ratio', 'Copper_Albumin_Ratio', 'SGOT_Platelets_Ratio',
            'Prothrombin_Albumin_Ratio', 'Liver_Function_Score', 'Clinical_Complications_Score',
            'Bilirubin_Severe', 'Albumin_Low', 'Platelets_Low', 'Prothrombin_High',
            'Bilirubin_Log', 'Copper_Log', 'Alk_Phos_Log',
            'Ascites_Binary', 'Hepatomegaly_Binary', 'Spiders_Binary', 'Edema_Ordinal'
        ]
        
        # Add drug and sex dummies if they exist
        drug_cols = [col for col in df_encoded.columns if col.startswith('Drug_')]
        sex_cols = [col for col in df_encoded.columns if col.startswith('Sex_')]
        feature_columns.extend(drug_cols + sex_cols)
        
        # Select available features
        available_features = [col for col in feature_columns if col in df_encoded.columns]
        X = df_encoded[available_features].fillna(0)
        
        return X, available_features
    
    def train_model(self, data_path='data/liver_cirrhosis_clean.csv'):
        """Train the liver cirrhosis prediction model"""
        try:
            logger.info("Starting model training...")
            
            # Load data
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                logger.info(f"Loaded data with shape: {df.shape}")
            else:
                logger.error(f"Data file not found: {data_path}")
                return False
            
            # Prepare features
            X, feature_names = self.prepare_features(df)
            y = df['Stage']
            
            self.feature_names = feature_names
            logger.info(f"Prepared {len(feature_names)} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble model
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Train models
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_score = rf_model.score(X_test_scaled, y_test)
            gb_score = gb_model.score(X_test_scaled, y_test)
            
            # Select best model
            if rf_score >= gb_score:
                self.model = rf_model
                self.model_accuracy = rf_score
                logger.info(f"Selected Random Forest model with accuracy: {rf_score:.3f}")
            else:
                self.model = gb_model
                self.model_accuracy = gb_score
                logger.info(f"Selected Gradient Boosting model with accuracy: {gb_score:.3f}")
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, self.model.feature_importances_))
                # Get top 10 most important features
                self.feature_importance = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                )
            
            # Generate classification report
            y_pred = self.model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred)
            logger.info(f"Classification Report:\n{report}")
            
            self.model_trained = True
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, patient_data):
        """Make prediction for a single patient"""
        try:
            if not self.model_trained:
                logger.error("Model not trained yet")
                return None
            
            # Convert to DataFrame
            df_patient = pd.DataFrame([patient_data])
            
            # Prepare features
            X_patient, _ = self.prepare_features(df_patient)
            
            # Ensure all expected features are present
            missing_features = set(self.feature_names) - set(X_patient.columns)
            for feature in missing_features:
                X_patient[feature] = 0
            
            # Reorder columns to match training data
            X_patient = X_patient[self.feature_names]
            
            # Scale features
            X_patient_scaled = self.scaler.transform(X_patient)
            
            # Make prediction
            stage_pred = self.model.predict(X_patient_scaled)[0]
            stage_proba = self.model.predict_proba(X_patient_scaled)[0]
            
            # Get confidence (probability of predicted class)
            confidence = stage_proba[stage_pred - 1]  # Adjust for 0-indexed probabilities
            
            return {
                'stage': int(stage_pred),
                'confidence': float(confidence),
                'probabilities': {
                    'stage_1': float(stage_proba[0]),
                    'stage_2': float(stage_proba[1]),
                    'stage_3': float(stage_proba[2])
                },
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def save_model(self, model_path='model/liver_cirrhosis_model.pkl'):
        """Save trained model and preprocessing components"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'model_accuracy': self.model_accuracy
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path='model/liver_cirrhosis_model.pkl'):
        """Load trained model and preprocessing components"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.feature_importance = model_data.get('feature_importance', {})
                self.model_accuracy = model_data.get('model_accuracy', 0.0)
                self.model_trained = True
                
                logger.info(f"Model loaded successfully from {model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

# Initialize model
model = LiverCirrhosisModel()

# Try to load existing model, otherwise train new one
if not model.load_model():
    logger.info("No existing model found, training new model...")
    model.train_model()

@app.route('/')
def home():
    """Serve the main application page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Liver Cirrhosis Stage Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .method { color: #27ae60; font-weight: bold; }
            code { background: #34495e; color: white; padding: 2px 6px; border-radius: 3px; }
            .status { padding: 10px; border-radius: 5px; margin: 20px 0; }
            .status.success { background: #d5edda; border: 1px solid #c3e6cb; color: #155724; }
            .status.error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🫀 Liver Cirrhosis Stage Detection API</h1>
            
            <div class="status {{ 'success' if model_status else 'error' }}">
                <strong>Model Status:</strong> {{ model_message }}
                {% if model_accuracy > 0 %}
                <br><strong>Model Accuracy:</strong> {{ "%.1f" | format(model_accuracy * 100) }}%
                {% endif %}
            </div>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /predict</h3>
                <p>Predict liver cirrhosis stage based on patient data</p>
                <p><strong>Content-Type:</strong> application/json</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /model-info</h3>
                <p>Get information about the trained model</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /retrain</h3>
                <p>Retrain the model (admin only)</p>
            </div>
            
            <h2>Example Usage</h2>
            <pre><code>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": 45.2,
    "sex": "F",
    "drug": "D-penicillamine",
    "n_days": 1200,
    "ascites": "N",
    "hepatomegaly": "Y",
    "spiders": "N",
    "edema": "S",
    "bilirubin": 2.3,
    "cholesterol": 280,
    "albumin": 3.1,
    "copper": 85,
    "alk_phos": 1450,
    "sgot": 120,
    "tryglicerides": 180,
    "platelets": 220,
    "prothrombin": 11.8
  }'</code></pre>
            
            <h2>Frontend Application</h2>
            <p>Access the interactive web interface at <a href="/app">/app</a></p>
        </div>
    </body>
    </html>
    """, 
    model_status=model.model_trained,
    model_message="Model trained and ready" if model.model_trained else "Model not available",
    model_accuracy=model.model_accuracy
    )

@app.route('/app')
def app_interface():
    """Serve the frontend application"""
    # In a real deployment, you would serve the HTML file created earlier
    return "Frontend application would be served here. Please serve the HTML file separately."

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'age', 'sex', 'drug', 'n_days', 'ascites', 'hepatomegaly', 
            'spiders', 'edema', 'bilirubin', 'cholesterol', 'albumin', 
            'copper', 'alk_phos', 'sgot', 'tryglicerides', 'platelets', 'prothrombin'
        ]
        
        missing_fields = [field for field in required_fields if field not in patient_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Convert field names to match model expectations
        field_mapping = {
            'age': 'Age',
            'sex': 'Sex', 
            'drug': 'Drug',
            'n_days': 'N_Days',
            'ascites': 'Ascites',
            'hepatomegaly': 'Hepatomegaly',
            'spiders': 'Spiders',
            'edema': 'Edema',
            'bilirubin': 'Bilirubin',
            'cholesterol': 'Cholesterol',
            'albumin': 'Albumin',
            'copper': 'Copper',
            'alk_phos': 'Alk_Phos',
            'sgot': 'SGOT',
            'tryglicerides': 'Tryglicerides',
            'platelets': 'Platelets',
            'prothrombin': 'Prothrombin'
        }
        
        # Convert patient data
        model_data = {}
        for frontend_key, model_key in field_mapping.items():
            if frontend_key in patient_data:
                model_data[model_key] = patient_data[frontend_key]
        
        # Make prediction
        prediction = model.predict(model_data)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Log prediction for monitoring
        logger.info(f"Prediction made: Stage {prediction['stage']} with confidence {prediction['confidence']:.3f}")
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat(),
            'model_accuracy': model.model_accuracy
        })
        
    except BadRequest as e:
        return jsonify({'error': 'Invalid request format'}), 400
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    try:
        info = {
            'model_trained': model.model_trained,
            'model_accuracy': model.model_accuracy,
            'feature_count': len(model.feature_names),
            'feature_names': model.feature_names,
            'feature_importance': model.feature_importance,
            'model_type': type(model.model).__name__ if model.model else None,
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': 'Failed to get model information'}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with new data"""
    try:
        # In production, add authentication/authorization here
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != 'Bearer admin-token':
            return jsonify({'error': 'Unauthorized'}), 401
        
        logger.info("Starting model retraining...")
        
        success = model.train_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model retrained successfully',
                'new_accuracy': model.model_accuracy,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Model retraining failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        return jsonify({'error': 'Failed to retrain model'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_ready': model.model_trained,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)