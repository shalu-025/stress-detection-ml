"""
Multimodal stress prediction model
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class MultimodalStressPredictor:
    """Multimodal stress level predictor combining physiological, image, and audio data"""
    
    def __init__(self):
        """Initialize the multimodal predictor"""
        self.model = None
        self.scaler = StandardScaler()
        self.ml_model = None  # ML model for training
        self.feature_names = [
            'heart_rate', 'spo2', 'temperature', 'age', 'heart_rate_variability',
            'oxygen_saturation_risk', 'age_group', 'temperature_deviation',
            'image_score', 'audio_score'
        ]
        
        # Initialize with dummy model for testing
        self._load_model()
    
    def _load_model(self):
        """Load the multimodal stress prediction model"""
        try:
            # For testing purposes, we'll create a simple model
            # In a real implementation, this would load a pre-trained model
            self.model = self._create_dummy_model()
        except Exception as e:
            print(f"Warning: Could not load multimodal model: {e}")
            self.model = None
    
    def _create_dummy_model(self):
        """Create a dummy model for testing"""
        def predict(physiological, image_score, audio_score):
            # Simple rule-based prediction for testing
            features = []
            
            # Extract physiological features
            hr = physiological.get('heart_rate', 75)
            spo2 = physiological.get('spo2', 98)
            temp = physiological.get('temperature', 37.0)
            age = physiological.get('age', 30)
            hrv = physiological.get('heart_rate_variability', 0.5)
            o2_risk = physiological.get('oxygen_saturation_risk', 0.2)
            age_group = physiological.get('age_group', 1)
            temp_dev = physiological.get('temperature_deviation', 0.0)
            
            # Normalize features
            hr_norm = (hr - 60) / (100 - 60)  # Normalize to [0,1]
            spo2_norm = (spo2 - 95) / (100 - 95)  # Normalize to [0,1]
            temp_norm = (temp - 36.5) / (37.5 - 36.5)  # Normalize to [0,1]
            age_norm = (age - 18) / (80 - 18)  # Normalize to [0,1]
            
            # Calculate stress score based on multiple factors
            stress_factors = [
                0.2 * (1 - hr_norm),  # Higher HR = more stress
                0.15 * (1 - spo2_norm),  # Lower SpO2 = more stress
                0.1 * abs(temp_norm - 0.5),  # Temperature deviation
                0.1 * (1 - hrv),  # Lower HRV = more stress
                0.15 * o2_risk,  # Higher risk = more stress
                0.1 * image_score,  # Image stress score
                0.1 * audio_score,  # Audio stress score
                0.1 * age_norm  # Age factor
            ]
            
            total_stress = sum(stress_factors)
            stress_score = np.clip(total_stress, 0.0, 1.0)
            
            # Determine stress level
            if stress_score < 0.3:
                stress_level = 'low'
            elif stress_score < 0.6:
                stress_level = 'medium'
            else:
                stress_level = 'high'
            
            # Calculate confidence based on feature consistency
            feature_variance = np.var(stress_factors)
            confidence = max(0.5, 1.0 - feature_variance)
            
            return stress_level, confidence
        
        return predict
    
    def train(self, multimodal_data: List[Dict[str, Any]], labels: List[str]) -> Dict[str, float]:
        """Train the multimodal stress predictor"""
        try:
            # Extract features from multimodal data
            features = []
            numeric_labels = []
            
            for data, label in zip(multimodal_data, labels):
                # Extract physiological features
                physiological = data.get('physiological', {})
                image_score = data.get('image_score', 0.5)
                audio_score = data.get('audio_score', 0.5)
                
                # Create feature vector
                feature_vector = [
                    physiological.get('heart_rate', 75),
                    physiological.get('spo2', 98),
                    physiological.get('temperature', 37.0),
                    physiological.get('age', 30),
                    physiological.get('heart_rate_variability', 0.5),
                    physiological.get('oxygen_saturation_risk', 0.2),
                    physiological.get('age_group', 1),
                    physiological.get('temperature_deviation', 0.0),
                    image_score,
                    audio_score
                ]
                
                features.append(feature_vector)
                
                # Convert string labels to numeric
                if label == 'low':
                    numeric_labels.append(0.0)
                elif label == 'medium':
                    numeric_labels.append(0.5)
                else:  # high
                    numeric_labels.append(1.0)
            
            features = np.array(features)
            numeric_labels = np.array(numeric_labels)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train a Linear Regression model
            self.ml_model = LinearRegression()
            self.ml_model.fit(features_scaled, numeric_labels)
            
            # Calculate training metrics
            predictions = self.ml_model.predict(features_scaled)
            mse = mean_squared_error(numeric_labels, predictions)
            r2 = r2_score(numeric_labels, predictions)
            
            return {
                'mse': mse,
                'r2_score': r2,
                'n_samples': len(multimodal_data)
            }
            
        except Exception as e:
            print(f"Error training multimodal model: {e}")
            return {'mse': float('inf'), 'r2_score': 0.0, 'n_samples': 0}
    
    def evaluate(self, test_data: List[Dict[str, Any]], test_labels: List[str]) -> Dict[str, float]:
        """Evaluate the multimodal stress predictor"""
        try:
            if self.ml_model is None:
                return {'mse': float('inf'), 'r2_score': 0.0, 'n_samples': 0}
            
            # Extract features from test data
            features = []
            numeric_labels = []
            
            for data, label in zip(test_data, test_labels):
                # Extract physiological features
                physiological = data.get('physiological', {})
                image_score = data.get('image_score', 0.5)
                audio_score = data.get('audio_score', 0.5)
                
                # Create feature vector
                feature_vector = [
                    physiological.get('heart_rate', 75),
                    physiological.get('spo2', 98),
                    physiological.get('temperature', 37.0),
                    physiological.get('age', 30),
                    physiological.get('heart_rate_variability', 0.5),
                    physiological.get('oxygen_saturation_risk', 0.2),
                    physiological.get('age_group', 1),
                    physiological.get('temperature_deviation', 0.0),
                    image_score,
                    audio_score
                ]
                
                features.append(feature_vector)
                
                # Convert string labels to numeric
                if label == 'low':
                    numeric_labels.append(0.0)
                elif label == 'medium':
                    numeric_labels.append(0.5)
                else:  # high
                    numeric_labels.append(1.0)
            
            features = np.array(features)
            numeric_labels = np.array(numeric_labels)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.ml_model.predict(features_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(numeric_labels, predictions)
            r2 = r2_score(numeric_labels, predictions)
            
            return {
                'mse': mse,
                'r2_score': r2,
                'n_samples': len(test_data)
            }
            
        except Exception as e:
            print(f"Error evaluating multimodal model: {e}")
            return {'mse': float('inf'), 'r2_score': 0.0, 'n_samples': 0}
    
    def save(self, filepath: str):
        """Save the trained model"""
        try:
            if self.ml_model is not None:
                model_data = {
                    'model': self.ml_model,
                    'scaler': self.scaler
                }
                joblib.dump(model_data, filepath)
                print(f"Multimodal model saved to {filepath}")
            else:
                print("No trained model to save")
        except Exception as e:
            print(f"Error saving multimodal model: {e}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.ml_model = model_data['model']
                self.scaler = model_data['scaler']
                print(f"Multimodal model loaded from {filepath}")
            else:
                print(f"Model file not found: {filepath}")
        except Exception as e:
            print(f"Error loading multimodal model: {e}")
    
    def predict(self, physiological: Dict[str, float], 
                image_score: float = 0.5, 
                audio_score: float = 0.5) -> Tuple[str, float]:
        """Predict stress level from multimodal data"""
        if self.ml_model is not None:
            # Use trained ML model
            try:
                # Create feature vector
                feature_vector = [
                    physiological.get('heart_rate', 75),
                    physiological.get('spo2', 98),
                    physiological.get('temperature', 37.0),
                    physiological.get('age', 30),
                    physiological.get('heart_rate_variability', 0.5),
                    physiological.get('oxygen_saturation_risk', 0.2),
                    physiological.get('age_group', 1),
                    physiological.get('temperature_deviation', 0.0),
                    image_score,
                    audio_score
                ]
                
                # Scale features
                feature_scaled = self.scaler.transform([feature_vector])
                
                # Make prediction
                prediction = self.ml_model.predict(feature_scaled)[0]
                prediction = np.clip(prediction, 0.0, 1.0)
                
                # Convert to stress level
                if prediction < 0.3:
                    stress_level = 'low'
                elif prediction < 0.6:
                    stress_level = 'medium'
                else:
                    stress_level = 'high'
                
                # Calculate confidence (simplified)
                confidence = 0.8  # Placeholder confidence
                
                return stress_level, confidence
                
            except Exception as e:
                print(f"Error in ML prediction: {e}")
        
        # Fallback to dummy model
        if self.model is None:
            return 'medium', 0.5  # Default values if model not loaded
        
        try:
            stress_level, confidence = self.model(physiological, image_score, audio_score)
            return stress_level, confidence
            
        except Exception as e:
            print(f"Error in multimodal prediction: {e}")
            return 'medium', 0.5  # Default fallback

def create_sample_multimodal_data(num_samples: int = 10) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Create sample multimodal data for testing"""
    np.random.seed(42)  # For reproducible results
    
    data = []
    labels = []
    
    for i in range(num_samples):
        # Generate physiological data
        physiological = {
            'heart_rate': int(np.random.uniform(60, 100)),
            'spo2': float(np.random.uniform(95, 100)),
            'temperature': float(np.random.uniform(36.5, 37.5)),
            'age': int(np.random.uniform(18, 80)),
            'heart_rate_variability': float(np.random.uniform(0.1, 0.8)),
            'oxygen_saturation_risk': float(np.random.uniform(0.0, 0.5)),
            'temperature_deviation': float(np.random.uniform(-0.5, 0.5))
        }
        
        # Age group encoding
        age = physiological['age']
        if age < 25:
            physiological['age_group'] = 0
        elif age < 45:
            physiological['age_group'] = 1
        elif age < 65:
            physiological['age_group'] = 2
        else:
            physiological['age_group'] = 3
        
        # Generate image and audio scores
        image_score = float(np.random.uniform(0.0, 1.0))
        audio_score = float(np.random.uniform(0.0, 1.0))
        
        # Create multimodal record
        record = {
            'physiological': physiological,
            'image_score': image_score,
            'audio_score': audio_score
        }
        
        # Determine label based on combined scores
        combined_score = (image_score + audio_score) / 2
        if combined_score < 0.4:
            label = 'low'
        elif combined_score < 0.7:
            label = 'medium'
        else:
            label = 'high'
        
        data.append(record)
        labels.append(label)
    
    return data, labels 