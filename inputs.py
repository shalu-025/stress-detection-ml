"""
Input collection and processing module for stress detection system
"""

import json
import numpy as np
from typing import Dict, List, Any

def load_defaults() -> Dict[str, Any]:
    """Load default parameters from defaults.json"""
    try:
        with open('defaults.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default values if file doesn't exist
        return {
            'heart_rate_range': [60, 100],
            'spo2_range': [95, 100],
            'temperature_range': [36.5, 37.5],
            'age_range': [18, 80],
            'stress_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            }
        }

def generate_sample_data(num_samples: int = 10) -> List[Dict[str, Any]]:
    """Generate sample physiological data for testing"""
    np.random.seed(42)  # For reproducible results
    
    samples = []
    for i in range(num_samples):
        sample = {
            'heart_rate': int(np.random.uniform(60, 100)),
            'spo2': float(np.random.uniform(95, 100)),
            'temperature': float(np.random.uniform(36.5, 37.5)),
            'age': int(np.random.uniform(18, 80)),
            'gender': np.random.choice(['male', 'female']),
            'heart_rate_variability': float(np.random.uniform(0.1, 0.8)),
            'oxygen_saturation_risk': float(np.random.uniform(0.0, 0.5)),
            'temperature_deviation': float(np.random.uniform(-0.5, 0.5))
        }
        samples.append(sample)
    
    return samples

def collect_inputs(raw_inputs: Dict[str, Any]) -> Dict[str, float]:
    """Extract and normalize features from raw inputs"""
    features = {}
    
    # Basic physiological features
    features['heart_rate'] = float(raw_inputs.get('heart_rate', 75))
    features['spo2'] = float(raw_inputs.get('spo2', 98))
    features['temperature'] = float(raw_inputs.get('temperature', 37.0))
    features['age'] = float(raw_inputs.get('age', 30))
    
    # Derived features
    features['heart_rate_variability'] = float(raw_inputs.get('heart_rate_variability', 0.5))
    features['oxygen_saturation_risk'] = float(raw_inputs.get('oxygen_saturation_risk', 0.2))
    features['temperature_deviation'] = float(raw_inputs.get('temperature_deviation', 0.0))
    
    # Age group encoding
    age = features['age']
    if age < 25:
        features['age_group'] = 0
    elif age < 45:
        features['age_group'] = 1
    elif age < 65:
        features['age_group'] = 2
    else:
        features['age_group'] = 3
    
    # Gender encoding
    gender = raw_inputs.get('gender', 'male')
    features['gender_encoded'] = 1.0 if gender == 'male' else 0.0
    
    return features 