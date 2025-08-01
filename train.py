import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from inputs import generate_sample_data
from image_module import ImageStressDetector, create_sample_image_data
from audio_module import AudioStressDetector, create_sample_audio_data
from model import MultimodalStressPredictor, create_sample_multimodal_data

class TrainingPipeline:
    """
    Training pipeline for the multimodal stress detection system
    """
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize the training pipeline
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger("TrainingPipeline")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.image_detector = ImageStressDetector()
        self.audio_detector = AudioStressDetector()
        self.multimodal_predictor = MultimodalStressPredictor()
        
        # Training results
        self.training_results = {}
    
    def generate_training_data(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate comprehensive training data for all modalities
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            Dictionary containing training data for all modalities
        """
        self.logger.info(f"Generating {num_samples} training samples...")
        
        training_data = {}
        
        try:
            # Generate physiological data
            physiological_data = generate_sample_data(num_samples)
            training_data['physiological'] = physiological_data
            
            # Generate image data
            image_data, image_labels = create_sample_image_data(num_samples)
            training_data['image'] = {
                'data': image_data,
                'labels': image_labels
            }
            
            # Generate audio data
            audio_data, audio_labels = create_sample_audio_data(num_samples)
            training_data['audio'] = {
                'data': audio_data,
                'labels': audio_labels
            }
            
            # Generate multimodal data
            multimodal_data, multimodal_labels = create_sample_multimodal_data(num_samples)
            training_data['multimodal'] = {
                'data': multimodal_data,
                'labels': multimodal_labels
            }
            
            self.logger.info("Training data generation completed")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error generating training data: {str(e)}")
            return {}
    
    def train_image_model(self, training_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the image-based stress detection model
        
        Args:
            training_data: Training data dictionary
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            self.logger.info("Training image model...")
            
            if 'image' not in training_data:
                raise ValueError("Image training data not found")
            
            image_data = training_data['image']['data']
            image_labels = training_data['image']['labels']
            
            # Convert labels to stress scores (0-1)
            stress_scores = []
            for label in image_labels:
                if isinstance(label, str):
                    # Convert string labels to scores
                    if label == 'low':
                        stress_scores.append(0.2)
                    elif label == 'medium':
                        stress_scores.append(0.5)
                    elif label == 'high':
                        stress_scores.append(0.8)
                    else:
                        stress_scores.append(0.5)
                else:
                    stress_scores.append(float(label))
            
            # Train the model
            self.image_detector.train(image_data, stress_scores)
            
            # Evaluate on a subset
            test_size = min(100, len(image_data) // 5)
            if len(image_data) > test_size:
                test_data = image_data[:test_size]
                test_labels = stress_scores[:test_size]
                
                predictions = []
                for img in test_data:
                    pred = self.image_detector.predict_stress(img)
                    predictions.append(pred)
                
                # Calculate metrics
                mse = np.mean((np.array(predictions) - np.array(test_labels)) ** 2)
                mae = np.mean(np.abs(np.array(predictions) - np.array(test_labels)))
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
            else:
                metrics = {'mse': 0.0, 'mae': 0.0, 'rmse': 0.0}
            
            # Save model
            model_path = os.path.join(self.output_dir, "image_stress_model.pth")
            self.image_detector.save_model(model_path)
            
            self.logger.info(f"Image model training completed. MSE: {metrics['mse']:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training image model: {str(e)}")
            return {}
    
    def train_audio_model(self, training_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the audio-based stress detection model
        
        Args:
            training_data: Training data dictionary
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            self.logger.info("Training audio model...")
            
            if 'audio' not in training_data:
                raise ValueError("Audio training data not found")
            
            audio_data = training_data['audio']['data']
            audio_labels = training_data['audio']['labels']
            
            # Convert labels to stress scores (0-1)
            stress_scores = []
            for label in audio_labels:
                if isinstance(label, str):
                    # Convert string labels to scores
                    if label == 'low':
                        stress_scores.append(0.2)
                    elif label == 'medium':
                        stress_scores.append(0.5)
                    elif label == 'high':
                        stress_scores.append(0.8)
                    else:
                        stress_scores.append(0.5)
                else:
                    stress_scores.append(float(label))
            
            # Train the model
            self.audio_detector.train(audio_data, stress_scores)
            
            # Evaluate on a subset
            test_size = min(100, len(audio_data) // 5)
            if len(audio_data) > test_size:
                test_data = audio_data[:test_size]
                test_labels = stress_scores[:test_size]
                
                predictions = []
                for audio in test_data:
                    pred = self.audio_detector.predict_stress(audio)
                    predictions.append(pred)
                
                # Calculate metrics
                mse = np.mean((np.array(predictions) - np.array(test_labels)) ** 2)
                mae = np.mean(np.abs(np.array(predictions) - np.array(test_labels)))
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
            else:
                metrics = {'mse': 0.0, 'mae': 0.0, 'rmse': 0.0}
            
            # Save model
            model_path = os.path.join(self.output_dir, "audio_stress_model.pkl")
            self.audio_detector.save_model(model_path)
            
            self.logger.info(f"Audio model training completed. MSE: {metrics['mse']:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training audio model: {str(e)}")
            return {}
    
    def train_multimodal_model(self, training_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the multimodal fusion model
        
        Args:
            training_data: Training data dictionary
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            self.logger.info("Training multimodal model...")
            
            if 'multimodal' not in training_data:
                raise ValueError("Multimodal training data not found")
            
            multimodal_data = training_data['multimodal']['data']
            multimodal_labels = training_data['multimodal']['labels']
            
            # Convert string labels to numeric
            label_encoder = LabelEncoder()
            numeric_labels = label_encoder.fit_transform(multimodal_labels)
            
            # Train the model
            self.multimodal_predictor.train(multimodal_data, numeric_labels)
            
            # Evaluate on a subset
            test_size = min(200, len(multimodal_data) // 5)
            if len(multimodal_data) > test_size:
                test_data = multimodal_data[:test_size]
                test_labels = numeric_labels[:test_size]
                
                metrics = self.multimodal_predictor.evaluate(test_data, test_labels)
            else:
                metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            
            # Save model
            model_path = os.path.join(self.output_dir, "multimodal_stress_model.pkl")
            self.multimodal_predictor.save_model(model_path)
            
            self.logger.info(f"Multimodal model training completed. Accuracy: {metrics.get('accuracy', 0):.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training multimodal model: {str(e)}")
            return {}
    
    def run_full_training(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            Dictionary containing all training results
        """
        try:
            self.logger.info("Starting full training pipeline...")
            
            # Generate training data
            training_data = self.generate_training_data(num_samples)
            
            if not training_data:
                raise ValueError("Failed to generate training data")
            
            # Train individual models
            image_results = self.train_image_model(training_data)
            audio_results = self.train_audio_model(training_data)
            multimodal_results = self.train_multimodal_model(training_data)
            
            # Compile results
            self.training_results = {
                'image_model': image_results,
                'audio_model': audio_results,
                'multimodal_model': multimodal_results,
                'training_timestamp': datetime.now().isoformat(),
                'num_samples': num_samples
            }
            
            # Save training report
            self.save_training_report()
            
            self.logger.info("Full training pipeline completed successfully")
            return self.training_results
            
        except Exception as e:
            self.logger.error(f"Error in full training pipeline: {str(e)}")
            return {}
    
    def save_training_report(self):
        """Save training results to a report file"""
        try:
            report_path = os.path.join(self.output_dir, "training_report.json")
            
            with open(report_path, 'w') as f:
                import json
                json.dump(self.training_results, f, indent=2, default=str)
            
            self.logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving training report: {str(e)}")
    
    def create_training_visualizations(self):
        """Create visualizations of training results"""
        try:
            if not self.training_results:
                self.logger.warning("No training results to visualize")
                return
            
            # Create output directory for plots
            plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Model performance comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = ['Image Model', 'Audio Model', 'Multimodal Model']
            metrics = ['mse', 'mse', 'accuracy']  # Different metrics for different models
            
            values = []
            for i, model in enumerate(['image_model', 'audio_model', 'multimodal_model']):
                if model in self.training_results:
                    metric = metrics[i]
                    value = self.training_results[model].get(metric, 0)
                    values.append(value)
                else:
                    values.append(0)
            
            bars = ax.bar(models, values, color=['blue', 'green', 'red'])
            ax.set_title('Model Performance Comparison')
            ax.set_ylabel('Performance Metric')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "model_performance.png"))
            plt.close()
            
            self.logger.info("Training visualizations created")
            
        except Exception as e:
            self.logger.error(f"Error creating training visualizations: {str(e)}")
    
    def load_trained_models(self):
        """Load all trained models"""
        try:
            # Load image model
            image_model_path = os.path.join(self.output_dir, "image_stress_model.pth")
            if os.path.exists(image_model_path):
                self.image_detector.load_model(image_model_path)
                self.logger.info("Image model loaded successfully")
            
            # Load audio model
            audio_model_path = os.path.join(self.output_dir, "audio_stress_model.pkl")
            if os.path.exists(audio_model_path):
                self.audio_detector.load_model(audio_model_path)
                self.logger.info("Audio model loaded successfully")
            
            # Load multimodal model
            multimodal_model_path = os.path.join(self.output_dir, "multimodal_stress_model.pkl")
            if os.path.exists(multimodal_model_path):
                self.multimodal_predictor.load_model(multimodal_model_path)
                self.logger.info("Multimodal model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading trained models: {str(e)}")
    
    def evaluate_models(self, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate all trained models
        
        Args:
            test_data: Optional test data, will generate if not provided
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            if test_data is None:
                # Generate test data
                test_data = self.generate_training_data(100)
            
            evaluation_results = {}
            
            # Evaluate image model
            if 'image' in test_data:
                image_metrics = self.evaluate_image_model(test_data['image'])
                evaluation_results['image_model'] = image_metrics
            
            # Evaluate audio model
            if 'audio' in test_data:
                audio_metrics = self.evaluate_audio_model(test_data['audio'])
                evaluation_results['audio_model'] = audio_metrics
            
            # Evaluate multimodal model
            if 'multimodal' in test_data:
                multimodal_metrics = self.evaluate_multimodal_model(test_data['multimodal'])
                evaluation_results['multimodal_model'] = multimodal_metrics
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            return {}
    
    def evaluate_image_model(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate image model performance"""
        try:
            predictions = []
            true_labels = []
            
            for img, label in zip(test_data['data'], test_data['labels']):
                pred = self.image_detector.predict_stress(img)
                predictions.append(pred)
                true_labels.append(float(label))
            
            # Calculate metrics
            mse = np.mean((np.array(predictions) - np.array(true_labels)) ** 2)
            mae = np.mean(np.abs(np.array(predictions) - np.array(true_labels)))
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating image model: {str(e)}")
            return {}
    
    def evaluate_audio_model(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate audio model performance"""
        try:
            predictions = []
            true_labels = []
            
            for audio, label in zip(test_data['data'], test_data['labels']):
                pred = self.audio_detector.predict_stress(audio)
                predictions.append(pred)
                true_labels.append(float(label))
            
            # Calculate metrics
            mse = np.mean((np.array(predictions) - np.array(true_labels)) ** 2)
            mae = np.mean(np.abs(np.array(predictions) - np.array(true_labels)))
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating audio model: {str(e)}")
            return {}
    
    def evaluate_multimodal_model(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate multimodal model performance"""
        try:
            # Use the model's built-in evaluation
            return self.multimodal_predictor.evaluate(test_data['data'], test_data['labels'])
            
        except Exception as e:
            self.logger.error(f"Error evaluating multimodal model: {str(e)}")
            return {}

def main():
    """Main training script"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline()
    
    # Run full training
    results = pipeline.run_full_training(num_samples=500)
    
    if results:
        print("Training completed successfully!")
        print("Results:", results)
        
        # Create visualizations
        pipeline.create_training_visualizations()
        
        # Evaluate models
        evaluation = pipeline.evaluate_models()
        print("Evaluation results:", evaluation)
    else:
        print("Training failed!")

if __name__ == "__main__":
    main() 