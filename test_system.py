#!/usr/bin/env python3
"""
Test script for the multimodal stress detection system
"""

import sys
import os
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_inputs():
    """Test input collection and processing"""
    print("Testing input collection...")
    
    try:
        from inputs import collect_inputs, load_defaults, generate_sample_data
        
        # Test loading defaults
        defaults = load_defaults()
        print(f"✓ Defaults loaded: {len(defaults)} parameters")
        
        # Test sample data generation
        sample_data = generate_sample_data(5)
        print(f"✓ Sample data generated: {len(sample_data)} records")
        
        # Test input collection
        test_inputs = {
            'heart_rate': 85,
            'spo2': 96,
            'age': 30,
            'gender': 'male'
        }
        
        features = collect_inputs(test_inputs)
        print(f"✓ Features extracted: {len(features)} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Input test failed: {str(e)}")
        return False

def test_image_module():
    """Test image analysis module"""
    print("Testing image analysis...")
    
    try:
        from image_module import ImageStressDetector, create_sample_image_data
        import numpy as np
        
        # Test detector initialization
        detector = ImageStressDetector()
        print("✓ Image detector initialized")
        
        # Test sample data generation
        images, labels = create_sample_image_data(3)
        print(f"✓ Sample images generated: {len(images)} images")
        
        # Test prediction
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        stress_score = detector.predict_stress(sample_image)
        print(f"✓ Stress prediction: {stress_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Image test failed: {str(e)}")
        return False

def test_audio_module():
    """Test audio analysis module"""
    print("Testing audio analysis...")
    
    try:
        from audio_module import AudioStressDetector, create_sample_audio_data
        import numpy as np
        
        # Test detector initialization
        detector = AudioStressDetector()
        print("✓ Audio detector initialized")
        
        # Test sample data generation
        audio_data, labels = create_sample_audio_data(3)
        print(f"✓ Sample audio generated: {len(audio_data)} samples")
        
        # Test prediction
        sample_audio = np.random.randn(22050) * 0.1  # 1 second
        stress_score = detector.predict_stress(sample_audio)
        print(f"✓ Stress prediction: {stress_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio test failed: {str(e)}")
        return False

def test_multimodal_model():
    """Test multimodal fusion model"""
    print("Testing multimodal model...")
    
    try:
        from model import MultimodalStressPredictor, create_sample_multimodal_data
        
        # Test predictor initialization
        predictor = MultimodalStressPredictor()
        print("✓ Multimodal predictor initialized")
        
        # Test sample data generation
        multimodal_data, labels = create_sample_multimodal_data(5)
        print(f"✓ Sample multimodal data generated: {len(multimodal_data)} records")
        
        # Test prediction
        test_physiological = {
            'heart_rate': 85,
            'spo2': 96,
            'age': 30,
            'temperature': 37.2,
            'heart_rate_variability': 0.4,
            'oxygen_saturation_risk': 0.3,
            'age_group': 1,
            'temperature_deviation': 0.2
        }
        
        stress_level, confidence = predictor.predict(
            physiological=test_physiological,
            image_score=0.6,
            audio_score=0.7
        )
        
        print(f"✓ Multimodal prediction: {stress_level} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Multimodal test failed: {str(e)}")
        return False

def test_utils():
    """Test utility functions"""
    print("Testing utilities...")
    
    try:
        from utils import setup_logging, generate_session_summary, validate_input_data
        
        # Test logging setup
        logger = setup_logging()
        print("✓ Logging setup completed")
        
        # Test session summary
        test_sessions = [
            {
                'timestamp': datetime.now(),
                'final_stress_level': 'medium',
                'confidence': 0.75,
                'physiological': {'heart_rate': 85, 'spo2': 96}
            }
        ]
        
        summary = generate_session_summary(test_sessions)
        print(f"✓ Session summary generated: {len(summary)} metrics")
        
        # Test input validation
        test_input = {'heart_rate': 85, 'spo2': 96, 'age': 30}
        is_valid = validate_input_data(test_input)
        print(f"✓ Input validation: {'passed' if is_valid else 'failed'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Utils test failed: {str(e)}")
        return False

def test_training_pipeline():
    """Test training pipeline"""
    print("Testing training pipeline...")
    
    try:
        from train import TrainingPipeline
        
        # Test pipeline initialization
        pipeline = TrainingPipeline()
        print("✓ Training pipeline initialized")
        
        # Test data generation
        training_data = pipeline.generate_training_data(10)
        print(f"✓ Training data generated: {len(training_data)} modalities")
        
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧠 Multimodal Stress Detection System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Input Collection", test_inputs),
        ("Image Analysis", test_image_module),
        ("Audio Analysis", test_audio_module),
        ("Multimodal Model", test_multimodal_model),
        ("Utilities", test_utils),
        ("Training Pipeline", test_training_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} - PASSED")
        else:
            print(f"❌ {test_name} - FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nTo run the application:")
        print("  streamlit run app.py")
        print("\nTo train models:")
        print("  python train.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 