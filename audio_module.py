"""
Audio analysis module for stress detection system
"""

import numpy as np
import librosa
from typing import Tuple, List
from scipy import signal

class AudioStressDetector:
    """Detect stress levels from audio signals"""
    
    def __init__(self):
        """Initialize the audio stress detector"""
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
            print(f"Warning: Could not load audio model: {e}")
            self.model_loaded = False
    
    def _create_dummy_extractor(self):
        """Create a dummy feature extractor for testing"""
        def extractor(audio):
            # Simple audio feature extraction
            # In a real implementation, this would use librosa or similar
            
            # Ensure audio is 1D
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Basic audio features
            features = []
            
            # RMS energy
            rms = np.sqrt(np.mean(audio**2))
            features.append(rms)
            
            # Spectral centroid
            if len(audio) > 1024:
                # Compute FFT
                fft = np.fft.fft(audio[:1024])
                magnitude = np.abs(fft)
                freqs = np.fft.fftfreq(len(audio[:1024]))
                
                # Spectral centroid
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                features.append(abs(centroid))
            else:
                features.append(0.0)
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
            features.append(zero_crossings / len(audio))
            
            # Spectral rolloff
            if len(audio) > 1024:
                rolloff = np.percentile(magnitude, 85)
                features.append(rolloff)
            else:
                features.append(0.0)
            
            return np.array(features)
        
        return extractor
    
    def _create_dummy_classifier(self):
        """Create a dummy classifier for testing"""
        def classifier(features):
            # Simple rule-based classifier for testing
            rms = features[0]
            centroid = features[1]
            zero_crossing_rate = features[2]
            
            # Simple stress scoring based on audio characteristics
            # Higher energy, higher frequency content, and more zero crossings
            # might indicate stress
            stress_score = (
                0.3 * np.clip(rms / 0.5, 0, 1) +
                0.4 * np.clip(centroid / 0.5, 0, 1) +
                0.3 * np.clip(zero_crossing_rate / 0.1, 0, 1)
            )
            
            return np.clip(stress_score, 0.0, 1.0)
        
        return classifier
    
    def predict_stress(self, audio: np.ndarray) -> float:
        """Predict stress level from audio"""
        if not self.model_loaded:
            return 0.5  # Default value if model not loaded
        
        try:
            # Ensure audio is in correct format
            if audio.dtype != np.float64:
                audio = audio.astype(np.float64)
            
            # Extract features
            features = self.feature_extractor(audio)
            
            # Predict stress level
            stress_score = self.classifier(features)
            
            return float(stress_score)
            
        except Exception as e:
            print(f"Error in stress prediction: {e}")
            return 0.5  # Default fallback

def create_sample_audio_data(num_samples: int = 5) -> Tuple[List[np.ndarray], List[int]]:
    """Create sample audio data for testing"""
    np.random.seed(42)  # For reproducible results
    
    audio_data = []
    labels = []
    
    sample_rate = 22050  # Standard sample rate
    
    for i in range(num_samples):
        # Generate 1 second of audio
        duration = 1.0
        samples = int(sample_rate * duration)
        
        if i % 2 == 0:
            # "Stressed" audio: higher frequency, more noise
            # Generate higher frequency content
            t = np.linspace(0, duration, samples)
            audio = (
                0.3 * np.sin(2 * np.pi * 800 * t) +  # Higher frequency
                0.2 * np.sin(2 * np.pi * 1200 * t) + # Even higher frequency
                0.1 * np.random.randn(samples)  # Noise
            )
            label = 1  # Stressed
        else:
            # "Calm" audio: lower frequency, less noise
            t = np.linspace(0, duration, samples)
            audio = (
                0.5 * np.sin(2 * np.pi * 200 * t) +  # Lower frequency
                0.2 * np.sin(2 * np.pi * 400 * t) +  # Medium frequency
                0.05 * np.random.randn(samples)  # Less noise
            )
            label = 0  # Not stressed
        
        audio_data.append(audio)
        labels.append(label)
    
    return audio_data, labels 