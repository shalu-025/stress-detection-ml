"""
Image analysis module for stress detection system
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Any
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class ImageStressDetector:
    """Detect stress levels from facial images"""
    
    def __init__(self):
        """Initialize the image stress detector"""
        self.model_loaded = False
        self.feature_extractor = None
        self.classifier = None
        self.ml_model = None  # ML model for training
        
        # Initialize with dummy model for testing
        self._load_model()
    
    def _load_model(self):
        """Load the stress detection model"""
        try:
            # For testing purposes, we'll create a simple feature extractor
            # In a real implementation, this would load a pre-trained model
            self.feature_extractor = self._create_dummy_extractor()
            self.classifier = self._create_dummy_classifier()
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load image model: {e}")
            self.model_loaded = False
    
    def _create_dummy_extractor(self):
        """Create a dummy feature extractor for testing"""
        def extractor(image):
            # Simple feature extraction: convert to grayscale and compute histogram
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Compute histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            # Return a fixed-size feature vector
            features = np.concatenate([
                hist[:64],  # First 64 histogram bins
                [np.mean(gray), np.std(gray), np.median(gray)]  # Basic statistics
            ])
            
            return features
        
        return extractor
    
    def _create_dummy_classifier(self):
        """Create a dummy classifier for testing"""
        def classifier(features):
            # Simple rule-based classifier for testing
            mean_intensity = features[-3]  # Mean intensity from features
            std_intensity = features[-2]   # Std intensity from features
            
            # Simple stress scoring based on image characteristics
            # Higher contrast and lower brightness might indicate stress
            stress_score = (std_intensity / 255.0) * (1.0 - mean_intensity / 255.0)
            return np.clip(stress_score, 0.0, 1.0)
        
        return classifier
    
    def train(self, images: List[np.ndarray], labels: List[float]) -> Dict[str, float]:
        """Train the image stress detector"""
        try:
            # Extract features from all images
            features = []
            for image in images:
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                feature_vector = self.feature_extractor(image)
                features.append(feature_vector)
            
            features = np.array(features)
            labels = np.array(labels)
            
            # Train a Random Forest regressor
            self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.ml_model.fit(features, labels)
            
            # Calculate training metrics
            predictions = self.ml_model.predict(features)
            mse = mean_squared_error(labels, predictions)
            r2 = r2_score(labels, predictions)
            
            return {
                'mse': mse,
                'r2_score': r2,
                'n_samples': len(images)
            }
            
        except Exception as e:
            print(f"Error training image model: {e}")
            return {'mse': float('inf'), 'r2_score': 0.0, 'n_samples': 0}
    
    def evaluate(self, test_images: List[np.ndarray], test_labels: List[float]) -> Dict[str, float]:
        """Evaluate the image stress detector"""
        try:
            if self.ml_model is None:
                return {'mse': float('inf'), 'r2_score': 0.0, 'n_samples': 0}
            
            # Extract features from test images
            features = []
            for image in test_images:
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                feature_vector = self.feature_extractor(image)
                features.append(feature_vector)
            
            features = np.array(features)
            test_labels = np.array(test_labels)
            
            # Make predictions
            predictions = self.ml_model.predict(features)
            
            # Calculate metrics
            mse = mean_squared_error(test_labels, predictions)
            r2 = r2_score(test_labels, predictions)
            
            return {
                'mse': mse,
                'r2_score': r2,
                'n_samples': len(test_images)
            }
            
        except Exception as e:
            print(f"Error evaluating image model: {e}")
            return {'mse': float('inf'), 'r2_score': 0.0, 'n_samples': 0}
    
    def save(self, filepath: str):
        """Save the trained model"""
        try:
            if self.ml_model is not None:
                joblib.dump(self.ml_model, filepath)
                print(f"Image model saved to {filepath}")
            else:
                print("No trained model to save")
        except Exception as e:
            print(f"Error saving image model: {e}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        try:
            if os.path.exists(filepath):
                self.ml_model = joblib.load(filepath)
                print(f"Image model loaded from {filepath}")
            else:
                print(f"Model file not found: {filepath}")
        except Exception as e:
            print(f"Error loading image model: {e}")
    
    def predict_stress(self, image: np.ndarray) -> float:
        """Predict stress level from image"""
        if self.ml_model is not None:
            # Use trained ML model
            try:
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                features = self.feature_extractor(image)
                prediction = self.ml_model.predict([features])[0]
                return float(np.clip(prediction, 0.0, 1.0))
            except Exception as e:
                print(f"Error in ML prediction: {e}")
        
        # Fallback to dummy model
        if not self.model_loaded:
            return 0.5  # Default value if model not loaded
        
        try:
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Extract features
            features = self.feature_extractor(image)
            
            # Predict stress level
            stress_score = self.classifier(features)
            
            return float(stress_score)
            
        except Exception as e:
            print(f"Error in stress prediction: {e}")
            return 0.5  # Default fallback

def create_sample_image_data(num_samples: int = 5) -> Tuple[List[np.ndarray], List[int]]:
    """Create sample image data for testing"""
    np.random.seed(42)  # For reproducible results
    
    images = []
    labels = []
    
    for i in range(num_samples):
        # Create random images with different characteristics
        if i % 2 == 0:
            # "Stressed" images: higher contrast, darker
            image = np.random.randint(0, 128, (224, 224, 3), dtype=np.uint8)
            label = 1  # Stressed
        else:
            # "Calm" images: lower contrast, brighter
            image = np.random.randint(128, 255, (224, 224, 3), dtype=np.uint8)
            label = 0  # Not stressed
        
        images.append(image)
        labels.append(label)
    
    return images, labels 