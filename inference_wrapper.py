"""
Inference wrapper for multimodal stress detection model
"""

import torch
import numpy as np
import cv2
import librosa
from typing import Dict, Any, Optional, Tuple
import os
import logging
from datetime import datetime

from multimodal_model import MultimodalStressModel, create_multimodal_model
from data_preprocessing import WESADPreprocessor, ANUStressDBPreprocessor, CREMADPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalInferenceWrapper:
    """Inference wrapper for multimodal stress detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = create_multimodal_model()
        
        # Initialize preprocessors
        self.physio_preprocessor = WESADPreprocessor()
        self.facial_preprocessor = ANUStressDBPreprocessor()
        self.audio_preprocessor = CREMADPreprocessor()
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No trained model found. Using untrained model.")
    
    def preprocess_physiological_data(self, ecg_data: np.ndarray, eda_data: np.ndarray,
                                    resp_data: np.ndarray, temp_data: np.ndarray,
                                    acc_data: np.ndarray) -> np.ndarray:
        """Preprocess physiological data for inference"""
        try:
            features = self.physio_preprocessor.extract_physiological_features(
                ecg_data, eda_data, resp_data, temp_data, acc_data
            )
            return features
        except Exception as e:
            logger.error(f"Error preprocessing physiological data: {e}")
            return np.zeros(25)
    
    def preprocess_facial_data(self, image: np.ndarray) -> np.ndarray:
        """Preprocess facial image for inference"""
        try:
            # Detect and crop face
            face_crop = self.facial_preprocessor.detect_and_crop_face(image)
            
            # Extract features
            features = self.facial_preprocessor.extract_facial_features(face_crop)
            return features
        except Exception as e:
            logger.error(f"Error preprocessing facial data: {e}")
            return np.zeros(70)
    
    def preprocess_audio_data(self, audio_data: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """Preprocess audio data for inference"""
        try:
            # Ensure audio is 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Extract features using librosa
            features = []
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            features.extend(mfccs_mean)
            features.extend(mfccs_std)
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_mean = np.mean(pitches[magnitudes > 0.1]) if np.any(magnitudes > 0.1) else 0.0
            pitch_std = np.std(pitches[magnitudes > 0.1]) if np.any(magnitudes > 0.1) else 0.0
            features.extend([pitch_mean, pitch_std])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            
            # Energy features
            rms = librosa.feature.rms(y=audio_data)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            # Additional features to reach 40 dimensions
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            
            features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
            
            # Pad or truncate to exactly 40 features
            if len(features) < 40:
                features.extend([0.0] * (40 - len(features)))
            elif len(features) > 40:
                features = features[:40]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preprocessing audio data: {e}")
            return np.zeros(40)
    
    def predict_stress(self, physio_data: Optional[Dict[str, np.ndarray]] = None,
                      facial_image: Optional[np.ndarray] = None,
                      audio_data: Optional[np.ndarray] = None,
                      audio_sample_rate: int = 22050) -> Dict[str, Any]:
        """
        Predict stress level from multimodal data
        
        Args:
            physio_data: Dictionary containing physiological data
                        {'ecg': array, 'eda': array, 'resp': array, 'temp': array, 'acc': array}
            facial_image: RGB image array (H, W, 3)
            audio_data: Audio array (samples,)
            audio_sample_rate: Sample rate of audio data
            
        Returns:
            Dictionary containing stress prediction and confidence
        """
        try:
            # Preprocess physiological data
            if physio_data is not None:
                physio_features = self.preprocess_physiological_data(
                    physio_data.get('ecg', np.array([])),
                    physio_data.get('eda', np.array([])),
                    physio_data.get('resp', np.array([])),
                    physio_data.get('temp', np.array([])),
                    physio_data.get('acc', np.array([]))
                )
            else:
                # Use default values if no physiological data provided
                physio_features = np.zeros(25)
            
            # Preprocess facial data
            if facial_image is not None:
                facial_features = self.preprocess_facial_data(facial_image)
            else:
                # Use default values if no facial data provided
                facial_features = np.zeros(70)
            
            # Preprocess audio data
            if audio_data is not None:
                audio_features = self.preprocess_audio_data(audio_data, audio_sample_rate)
            else:
                # Use default values if no audio data provided
                audio_features = np.zeros(40)
            
            # Make prediction
            prediction = self.model.predict_stress_level(
                physio_features, facial_features, audio_features
            )
            
            # Add metadata
            prediction['timestamp'] = datetime.now().isoformat()
            prediction['modalities_used'] = {
                'physiological': physio_data is not None,
                'facial': facial_image is not None,
                'audio': audio_data is not None
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in stress prediction: {e}")
            return {
                'stress_level': 'unknown',
                'stress_score': 0.5,
                'uncertainty': 1.0,
                'confidence': 0.0,
                'physio_score': 0.5,
                'facial_score': 0.5,
                'audio_score': 0.5,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def predict_from_files(self, physio_file: Optional[str] = None,
                          facial_file: Optional[str] = None,
                          audio_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict stress level from file inputs
        
        Args:
            physio_file: Path to physiological data file (CSV/JSON)
            facial_file: Path to facial image file (JPG/PNG)
            audio_file: Path to audio file (WAV/MP3)
            
        Returns:
            Dictionary containing stress prediction and confidence
        """
        try:
            # Load physiological data
            physio_data = None
            if physio_file and os.path.exists(physio_file):
                if physio_file.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(physio_file)
                    physio_data = {
                        'ecg': df.get('ecg', np.zeros(1000)).values,
                        'eda': df.get('eda', np.zeros(1000)).values,
                        'resp': df.get('resp', np.zeros(1000)).values,
                        'temp': df.get('temp', np.zeros(1000)).values,
                        'acc': df.get('acc', np.zeros((1000, 3))).values
                    }
                elif physio_file.endswith('.json'):
                    import json
                    with open(physio_file, 'r') as f:
                        data = json.load(f)
                    physio_data = data
            
            # Load facial image
            facial_image = None
            if facial_file and os.path.exists(facial_file):
                facial_image = cv2.imread(facial_file)
                if facial_image is not None:
                    facial_image = cv2.cvtColor(facial_image, cv2.COLOR_BGR2RGB)
            
            # Load audio data
            audio_data = None
            if audio_file and os.path.exists(audio_file):
                audio_data, sample_rate = librosa.load(audio_file, sr=22050)
            else:
                sample_rate = 22050
            
            # Make prediction
            return self.predict_stress(physio_data, facial_image, audio_data, sample_rate)
            
        except Exception as e:
            logger.error(f"Error predicting from files: {e}")
            return {
                'stress_level': 'unknown',
                'stress_score': 0.5,
                'uncertainty': 1.0,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_modality_contributions(self, physio_data: np.ndarray, facial_data: np.ndarray,
                                 audio_data: np.ndarray) -> Dict[str, float]:
        """Get contribution of each modality to the final prediction"""
        try:
            # Get individual modality predictions
            with torch.no_grad():
                physio_tensor = torch.FloatTensor(physio_data).unsqueeze(0)
                facial_tensor = torch.FloatTensor(facial_data).unsqueeze(0)
                audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
                
                outputs = self.model(physio_tensor, facial_tensor, audio_tensor)
                
                return {
                    'physio_contribution': outputs['physio_stress'].item(),
                    'facial_contribution': outputs['facial_stress'].item(),
                    'audio_contribution': outputs['audio_stress'].item(),
                    'fused_prediction': outputs['fused_stress'].item()
                }
                
        except Exception as e:
            logger.error(f"Error getting modality contributions: {e}")
            return {
                'physio_contribution': 0.5,
                'facial_contribution': 0.5,
                'audio_contribution': 0.5,
                'fused_prediction': 0.5
            }
    
    def generate_diagnostic_report(self, prediction: Dict[str, Any]) -> str:
        """Generate a diagnostic report for the prediction"""
        report = f"""
Multimodal Stress Detection Report
================================
Timestamp: {prediction.get('timestamp', 'Unknown')}

Stress Level: {prediction.get('stress_level', 'Unknown')}
Stress Score: {prediction.get('stress_score', 0.0):.3f}
Confidence: {prediction.get('confidence', 0.0):.3f}
Uncertainty: {prediction.get('uncertainty', 0.0):.3f}

Modality Scores:
- Physiological: {prediction.get('physio_score', 0.0):.3f}
- Facial: {prediction.get('facial_score', 0.0):.3f}
- Audio: {prediction.get('audio_score', 0.0):.3f}

Modalities Used:
- Physiological: {prediction.get('modalities_used', {}).get('physiological', False)}
- Facial: {prediction.get('modalities_used', {}).get('facial', False)}
- Audio: {prediction.get('modalities_used', {}).get('audio', False)}

Recommendations:
"""
        
        stress_level = prediction.get('stress_level', 'unknown')
        if stress_level == 'low':
            report += "- Continue current healthy lifestyle\n- Monitor stress levels regularly\n"
        elif stress_level == 'medium':
            report += "- Consider stress management techniques\n- Take regular breaks\n- Practice relaxation exercises\n"
        elif stress_level == 'high':
            report += "- Immediate stress reduction recommended\n- Consider professional help\n- Prioritize self-care activities\n"
        else:
            report += "- Unable to determine stress level\n- Consider providing more data\n"
        
        if prediction.get('error'):
            report += f"\nError: {prediction['error']}"
        
        return report

def main():
    """Example usage of the inference wrapper"""
    # Initialize wrapper
    wrapper = MultimodalInferenceWrapper()
    
    # Example 1: Predict with simulated data
    print("Example 1: Predicting with simulated data")
    
    # Simulate physiological data
    physio_data = {
        'ecg': np.random.randn(1000) * 0.1 + np.sin(np.linspace(0, 10*np.pi, 1000)),
        'eda': np.random.randn(1000) * 0.05 + 0.5,
        'resp': np.random.randn(1000) * 0.1 + np.sin(np.linspace(0, 5*np.pi, 1000)),
        'temp': np.random.randn(1000) * 0.5 + 37.0,
        'acc': np.random.randn(1000, 3) * 0.1
    }
    
    # Simulate facial image
    facial_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Simulate audio data
    audio_data = np.random.randn(22050) * 0.1 + np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    
    # Make prediction
    prediction = wrapper.predict_stress(physio_data, facial_image, audio_data)
    
    # Generate report
    report = wrapper.generate_diagnostic_report(prediction)
    print(report)
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Predict with missing modalities
    print("Example 2: Predicting with only physiological data")
    
    prediction_partial = wrapper.predict_stress(physio_data=physio_data)
    report_partial = wrapper.generate_diagnostic_report(prediction_partial)
    print(report_partial)

if __name__ == "__main__":
    main() 