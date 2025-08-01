"""
Image analysis module for stress detection system
"""

import numpy as np
import cv2
from typing import Tuple, List
from PIL import Image

class ImageStressDetector:
    """Detect stress levels from facial images"""
    
    def __init__(self):
        """Initialize the image stress detector"""
        self.model_loaded = False
        self.feature_extractor = None
        self.classifier = None
        
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
    
    def predict_stress(self, image: np.ndarray) -> float:
        """Predict stress level from image"""
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