# Multimodal Stress Detection System

A comprehensive stress detection system that combines physiological, facial expression, and voice tone analysis to provide accurate stress level predictions.

## Features

### 🔬 Multimodal Analysis
- **Physiological Monitoring**: Heart rate, SpO2, age, gender, temperature
- **Facial Expression Analysis**: CNN-based stress detection from webcam images
- **Voice Tone Analysis**: Acoustic feature extraction for stress indicators
- **Ensemble Fusion**: Weighted combination of all modalities

### 📊 Progressive Input Collection
- Collects physiological inputs with default value fallbacks
- Handles missing modalities gracefully
- Real-time input validation and quality scoring

### 🎯 Smart Prediction
- Three-level stress classification (Low/Medium/High)
- Confidence scoring for predictions
- Personalized stress management suggestions

### 📈 Analytics & Reporting
- Session history tracking
- Stress level distribution charts
- Export capabilities (CSV/PDF)
- Real-time analytics dashboard

## Architecture

```
stress-detection-ml/
├── app.py                 # Main Streamlit application
├── inputs.py              # Input collection and validation
├── image_module.py        # Facial expression analysis
├── audio_module.py        # Voice tone analysis
├── model.py              # Multimodal fusion model
├── ui.py                 # Streamlit UI components
├── utils.py              # Utility functions
├── train.py              # Training pipeline
├── defaults.json         # Default physiological values
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stress-detection-ml
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Training Models

```bash
python train.py
```

This will:
- Generate sample training data
- Train image, audio, and multimodal models
- Save models to `models/` directory
- Generate performance visualizations

## System Components

### 1. Input Collection (`inputs.py`)
- **`collect_inputs()`**: Main function for gathering all inputs
- **`load_defaults()`**: Loads default values from JSON
- **`validate_physiological_inputs()`**: Validates and fills missing data
- **`extract_features_from_physiological()`**: Extracts stress-relevant features

### 2. Image Analysis (`image_module.py`)
- **`ImageStressDetector`**: CNN-based facial expression analysis
- Uses MobileNetV2 for feature extraction
- Transfer learning approach for stress classification
- Fallback to simple heuristics if CNN unavailable

### 3. Audio Analysis (`audio_module.py`)
- **`AudioStressDetector`**: Voice tone stress detection
- Extracts acoustic features (pitch, energy, MFCCs, etc.)
- Uses Random Forest classifier
- Comprehensive feature engineering

### 4. Multimodal Fusion (`model.py`)
- **`MultimodalStressPredictor`**: Ensemble model combining all modalities
- Weighted averaging of individual predictions
- Handles missing modalities gracefully
- Confidence scoring for predictions

### 5. Training Pipeline (`train.py`)
- **`TrainingPipeline`**: Complete training workflow
- Generates synthetic training data
- Trains all models with cross-validation
- Performance evaluation and visualization

## Data Flow

```
User Inputs → Validation → Feature Extraction → Model Prediction → Ensemble Fusion → Results
     ↓              ↓              ↓              ↓              ↓              ↓
Physiological   Image Data    Audio Data    Individual    Weighted      Stress Level
   Vitals      (Webcam)     (Microphone)   Predictions   Average      + Confidence
```

## Configuration

### Default Values (`defaults.json`)
```json
{
  "heart_rate": 75.0,
  "spo2": 98.0,
  "age": 30,
  "gender": "unknown",
  "temperature": 37.0,
  "blood_pressure_systolic": 120,
  "blood_pressure_diastolic": 80,
  "respiratory_rate": 16
}
```

### Modality Weights
- Physiological: 40%
- Image Analysis: 30%
- Audio Analysis: 30%

## API Reference

### Core Functions

#### `collect_inputs(user_inputs: Dict) -> Dict`
Collects and processes all inputs for stress detection.

**Parameters:**
- `user_inputs`: Dictionary containing user inputs

**Returns:**
- Dictionary with processed feature vector

#### `predict_stress(image/audio_data) -> float`
Predicts stress level from single modality.

**Parameters:**
- `image/audio_data`: Input data

**Returns:**
- Stress score (0-1)

#### `predict(physiological, image_score, audio_score) -> Tuple[str, float]`
Combines all modalities for final prediction.

**Returns:**
- Tuple of (stress_level, confidence)

## Training

### Data Generation
The system includes functions to generate synthetic training data:

```python
from inputs import generate_sample_data
from image_module import create_sample_image_data
from audio_module import create_sample_audio_data
from model import create_sample_multimodal_data

# Generate sample data
physiological_data = generate_sample_data(1000)
image_data, image_labels = create_sample_image_data(1000)
audio_data, audio_labels = create_sample_audio_data(1000)
multimodal_data, multimodal_labels = create_sample_multimodal_data(1000)
```

### Model Training
```python
from train import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_full_training(num_samples=1000)
```

## Performance Metrics

### Image Model
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Audio Model
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Multimodal Model
- Accuracy
- Precision
- Recall
- F1-Score

## File Structure

```
stress-detection-ml/
├── app.py                 # Main application
├── inputs.py              # Input processing
├── image_module.py        # Image analysis
├── audio_module.py        # Audio analysis
├── model.py              # Multimodal fusion
├── ui.py                 # UI components
├── utils.py              # Utilities
├── train.py              # Training pipeline
├── defaults.json         # Default values
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── models/              # Trained models
│   ├── image_stress_model.pth
│   ├── audio_stress_model.pkl
│   └── multimodal_stress_model.pkl
├── data/                # Session data
│   └── session_history.csv
└── logs/                # Log files
    └── stress_detection.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- PyTorch for deep learning capabilities
- Scikit-learn for machine learning algorithms
- Librosa for audio processing
- OpenCV for computer vision tasks

## Support

For issues and questions, please open an issue on the GitHub repository.
