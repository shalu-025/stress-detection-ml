# Multimodal Stress Detection System

A comprehensive stress detection system that combines physiological, facial, and audio modalities to predict stress levels with uncertainty estimation.

## Overview

This system implements a multimodal approach to stress detection using three key datasets:

1. **WESAD (Wearable Stress and Affect Detection)** - Physiological data (ECG, EDA, respiration, temperature, motion)
2. **ANUStressDB** - Facial RGB + thermal video/images for stress recognition
3. **CREMA-D** - Audio-visual emotional speech (audio modality for voice-based stress detection)

## Architecture

### Model Components

- **PhysiologicalEncoder**: Processes wearable sensor data (25 features)
- **FacialEncoder**: Analyzes facial expressions and features (70 features)
- **AudioEncoder**: Extracts voice characteristics (40 features)
- **CrossModalAttention**: Multi-head attention mechanism for modality fusion
- **MultimodalFusion**: Combines all modalities with uncertainty estimation

### Key Features

- **Cross-modal attention**: Learns relationships between different modalities
- **Uncertainty estimation**: Provides confidence scores for predictions
- **Modality dropout**: Handles missing modalities gracefully
- **Synthetic fusion strategy**: Combines single-modality datasets for training

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Setup

### 1. WESAD Dataset
```bash
# Download from UCI Machine Learning Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00442/wesad.zip
unzip wesad.zip
```

### 2. ANUStressDB
```bash
# Download from ResearchGate or contact authors
# Place in data/anustressdb/ directory
```

### 3. CREMA-D Dataset
```bash
# Download from Kaggle
kaggle datasets download -d ejlok1/cremad
unzip cremad.zip
```

## Usage

### 1. Data Preprocessing

```python
from data_preprocessing import MultimodalDataPreprocessor

# Initialize preprocessor
preprocessor = MultimodalDataPreprocessor()

# Preprocess all datasets
preprocessor.preprocess_all_datasets()
```

### 2. Training the Model

```python
from train_multimodal import MultimodalTrainingPipeline

# Initialize training pipeline
pipeline = MultimodalTrainingPipeline()

# Run full training
pipeline.run_full_training()
```

### 3. Inference

```python
from inference_wrapper import MultimodalInferenceWrapper

# Initialize wrapper
wrapper = MultimodalInferenceWrapper("multimodal_models/best_model.pth")

# Predict with all modalities
prediction = wrapper.predict_stress(
    physio_data={
        'ecg': ecg_array,
        'eda': eda_array,
        'resp': resp_array,
        'temp': temp_array,
        'acc': acc_array
    },
    facial_image=facial_image_array,
    audio_data=audio_array
)

print(f"Stress Level: {prediction['stress_level']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

### 4. File-based Inference

```python
# Predict from files
prediction = wrapper.predict_from_files(
    physio_file="data/physio_data.csv",
    facial_file="data/face.jpg",
    audio_file="data/voice.wav"
)
```

## Model Architecture Details

### Physiological Encoder
- Input: 25 features (ECG, EDA, respiration, temperature, acceleration)
- Architecture: MLP with dropout
- Output: 64-dimensional latent representation

### Facial Encoder
- Input: 70 features (histogram, statistical, edge features)
- Architecture: Deeper MLP for complex facial patterns
- Output: 64-dimensional latent representation

### Audio Encoder
- Input: 40 features (MFCCs, pitch, spectral, energy features)
- Architecture: MLP with dropout
- Output: 64-dimensional latent representation

### Cross-Modal Attention
- Multi-head attention mechanism
- Learns relationships between modalities
- 4 attention heads, 16-dimensional head size

### Fusion Module
- Combines all modality representations
- Outputs stress score and uncertainty
- 128-dimensional fusion layers

## Training Process

1. **Data Preprocessing**: Extract features from each modality
2. **Synthetic Fusion**: Combine single-modality datasets
3. **Model Training**: Train with cross-modal attention
4. **Validation**: Evaluate on held-out data
5. **Uncertainty Estimation**: Learn prediction confidence

## Evaluation Metrics

- **Stress Classification Accuracy**: Binary stress vs. no-stress
- **Stress Score Regression**: MSE for continuous stress levels
- **Uncertainty Calibration**: Confidence vs. accuracy correlation
- **Modality Ablation**: Performance with missing modalities

## Output Format

```json
{
    "stress_level": "medium",
    "stress_score": 0.45,
    "uncertainty": 0.12,
    "confidence": 0.88,
    "physio_score": 0.42,
    "facial_score": 0.51,
    "audio_score": 0.38,
    "timestamp": "2024-01-15T10:30:00",
    "modalities_used": {
        "physiological": true,
        "facial": true,
        "audio": true
    }
}
```

## Limitations and Considerations

### Dataset Limitations
- **Cross-dataset subjects**: Different subjects across datasets
- **Synthetic fusion**: No real same-person multimodal data
- **Domain shift**: Different recording conditions

### Technical Limitations
- **Simulated data**: Currently uses synthetic data for demonstration
- **Feature extraction**: Basic features, could be enhanced with deep learning
- **Fusion strategy**: Simple attention mechanism, could use more sophisticated methods

### Recommendations for Real Deployment
1. **Collect real multimodal data**: Same subjects across modalities
2. **Enhance feature extraction**: Use pre-trained models (ResNet, VGG for faces)
3. **Advanced fusion**: Implement transformer-based fusion
4. **Calibration**: Fine-tune on target population
5. **Privacy**: Ensure data privacy and consent

## File Structure

```
multimodal_stress_detection/
├── data_preprocessing.py      # Dataset preprocessing
├── multimodal_model.py        # Model architecture
├── train_multimodal.py        # Training pipeline
├── inference_wrapper.py       # Inference interface
├── requirements.txt           # Dependencies
├── README_multimodal.md       # This file
├── data/                      # Dataset directories
│   ├── wesad/
│   ├── anustressdb/
│   └── cremad/
├── processed_data/            # Preprocessed features
├── multimodal_models/         # Trained models
└── examples/                  # Usage examples
```

## Citation

If you use this system, please cite the original datasets:

```bibtex
@article{schmidt2018introducing,
  title={Introducing WESAD, a multimodal dataset for wearable stress and affect detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  journal={Proceedings of the 20th ACM international conference on multimodal interaction},
  year={2018}
}

@article{sharma2019thermal,
  title={Thermal spatio-temporal data for stress recognition},
  author={Sharma, Anjali and others},
  journal={SpringerOpen},
  year={2019}
}

@article{cao2014crema,
  title={CREMA-D: Crowd-sourced emotional multimodal actors dataset},
  author={Cao, Houwei and others},
  journal={IEEE transactions on affective computing},
  year={2014}
}
```

## License

This project is for research purposes. Please ensure compliance with dataset licenses and ethical guidelines for stress detection applications.

## Contact

For questions or contributions, please open an issue on the repository. 