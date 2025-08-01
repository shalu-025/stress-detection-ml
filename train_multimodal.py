"""
Training script for multimodal stress detection model
Uses WESAD, ANUStressDB, and CREMA-D datasets
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import MultimodalDataPreprocessor
from multimodal_model import MultimodalStressModel, MultimodalTrainer, create_multimodal_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalTrainingPipeline:
    """Complete training pipeline for multimodal stress detection"""
    
    def __init__(self, output_dir: str = "multimodal_models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data preprocessor
        self.data_preprocessor = MultimodalDataPreprocessor()
        
        # Initialize model
        self.model = create_multimodal_model()
        
        # Initialize trainer
        self.trainer = MultimodalTrainer(self.model)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
    def preprocess_datasets(self):
        """Preprocess all three datasets"""
        logger.info("Preprocessing datasets...")
        self.data_preprocessor.preprocess_all_datasets()
        logger.info("Dataset preprocessing completed!")
        
    def load_processed_data(self) -> Dict[str, Any]:
        """Load processed data from files"""
        logger.info("Loading processed data...")
        
        # Find the latest processed data files
        processed_dir = "processed_data"
        if not os.path.exists(processed_dir):
            logger.error("Processed data directory not found. Run preprocessing first.")
            return {}
        
        # Load the most recent files
        files = os.listdir(processed_dir)
        wesad_files = [f for f in files if f.startswith("wesad_processed")]
        anustress_files = [f for f in files if f.startswith("anustress_processed")]
        cremad_files = [f for f in files if f.startswith("cremad_processed")]
        
        if not wesad_files or not anustress_files or not cremad_files:
            logger.error("Processed data files not found. Run preprocessing first.")
            return {}
        
        # Load data
        wesad_data = np.load(os.path.join(processed_dir, wesad_files[-1]))
        anustress_data = np.load(os.path.join(processed_dir, anustress_files[-1]))
        cremad_data = np.load(os.path.join(processed_dir, cremad_files[-1]))
        
        return {
            'wesad': {
                'features': wesad_data['features'],
                'labels': wesad_data['labels']
            },
            'anustress': {
                'features': anustress_data['features'],
                'labels': anustress_data['labels']
            },
            'cremad': {
                'features': cremad_data['features'],
                'labels': cremad_data['labels']
            }
        }
    
    def create_synthetic_multimodal_data(self, data: Dict[str, Any], num_samples: int = 1000) -> Dict[str, Any]:
        """Create synthetic multimodal data by combining single-modality data"""
        logger.info("Creating synthetic multimodal data...")
        
        # Get data from each modality
        wesad_features = data['wesad']['features']
        anustress_features = data['anustress']['features']
        cremad_features = data['cremad']['features']
        
        # Create synthetic multimodal samples
        multimodal_data = {
            'physio_data': [],
            'facial_data': [],
            'audio_data': [],
            'labels': []
        }
        
        for i in range(num_samples):
            # Sample from each modality
            physio_idx = i % len(wesad_features)
            facial_idx = i % len(anustress_features)
            audio_idx = i % len(cremad_features)
            
            # Get features
            physio_feat = wesad_features[physio_idx]
            facial_feat = anustress_features[facial_idx]
            audio_feat = cremad_features[audio_idx]
            
            # Create synthetic label based on combination
            physio_label = data['wesad']['labels'][physio_idx]
            facial_label = data['anustress']['labels'][facial_idx]
            audio_label = data['cremad']['labels'][audio_idx]
            
            # Combine labels (simple average for now)
            combined_label = (physio_label + facial_label + audio_label) / 3
            
            # Store data
            multimodal_data['physio_data'].append(physio_feat)
            multimodal_data['facial_data'].append(facial_feat)
            multimodal_data['audio_data'].append(audio_feat)
            multimodal_data['labels'].append(combined_label)
        
        # Convert to numpy arrays
        for key in multimodal_data:
            multimodal_data[key] = np.array(multimodal_data[key])
        
        logger.info(f"Created {num_samples} synthetic multimodal samples")
        return multimodal_data
    
    def prepare_data_loaders(self, data: Dict[str, Any], batch_size: int = 32) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Prepare data loaders for training"""
        logger.info("Preparing data loaders...")
        
        # Create synthetic multimodal data
        multimodal_data = self.create_synthetic_multimodal_data(data)
        
        # Split into train/validation
        total_samples = len(multimodal_data['labels'])
        train_size = int(0.8 * total_samples)
        
        # Create train data
        train_data = {
            'physio': multimodal_data['physio_data'][:train_size],
            'facial': multimodal_data['facial_data'][:train_size],
            'audio': multimodal_data['audio_data'][:train_size],
            'labels': multimodal_data['labels'][:train_size]
        }
        
        # Create validation data
        val_data = {
            'physio': multimodal_data['physio_data'][train_size:],
            'facial': multimodal_data['facial_data'][train_size:],
            'audio': multimodal_data['audio_data'][train_size:],
            'labels': multimodal_data['labels'][train_size:]
        }
        
        # Create datasets
        train_dataset = MultimodalDataset(train_data)
        val_dataset = MultimodalDataset(val_data)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_loader, val_loader
    
    def train_model(self, train_loader: torch.utils.data.DataLoader, 
                   val_loader: torch.utils.data.DataLoader, 
                   num_epochs: int = 50) -> Dict[str, List[float]]:
        """Train the multimodal model"""
        logger.info("Starting model training...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_losses = []
            train_accuracies = []
            
            self.model.train()
            for batch in train_loader:
                physio_data, facial_data, audio_data, labels = batch
                
                # Training step
                loss_dict = self.trainer.train_step(physio_data, facial_data, audio_data, labels)
                train_losses.append(loss_dict['total_loss'])
                
                # Calculate accuracy
                with torch.no_grad():
                    outputs = self.model(physio_data, facial_data, audio_data)
                    predictions = (outputs['fused_stress'] > 0.5).float()
                    accuracy = (predictions == labels).float().mean().item()
                    train_accuracies.append(accuracy)
            
            # Validation phase
            val_losses = []
            val_accuracies = []
            
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    physio_data, facial_data, audio_data, labels = batch
                    
                    # Evaluation
                    metrics = self.trainer.evaluate(physio_data, facial_data, audio_data, labels)
                    val_losses.append(metrics['stress_loss'])
                    val_accuracies.append(metrics['accuracy'])
            
            # Calculate averages
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accuracies)
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_accuracy'].append(avg_train_acc)
            self.training_history['val_accuracy'].append(avg_val_acc)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.model.save_model(os.path.join(self.output_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training completed!")
        return self.training_history
    
    def create_training_visualizations(self):
        """Create training visualizations"""
        logger.info("Creating training visualizations...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Loss plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss')
        plt.plot(self.training_history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['train_accuracy'], label='Train Accuracy')
        plt.plot(self.training_history['val_accuracy'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_curves.png"))
        plt.close()
        
        logger.info("Training visualizations created!")
    
    def save_training_report(self):
        """Save training report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'model_architecture': {
                'physio_input_dim': 25,
                'facial_input_dim': 70,
                'audio_input_dim': 40,
                'latent_dim': 64,
                'fusion_dim': 128
            },
            'training_history': self.training_history,
            'final_metrics': {
                'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
                'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
                'final_train_acc': self.training_history['train_accuracy'][-1] if self.training_history['train_accuracy'] else None,
                'final_val_acc': self.training_history['val_accuracy'][-1] if self.training_history['val_accuracy'] else None
            }
        }
        
        report_path = os.path.join(self.output_dir, f"training_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {report_path}")
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting full multimodal training pipeline...")
        
        # Step 1: Preprocess datasets
        self.preprocess_datasets()
        
        # Step 2: Load processed data
        data = self.load_processed_data()
        if not data:
            logger.error("Failed to load processed data")
            return
        
        # Step 3: Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(data)
        
        # Step 4: Train model
        training_history = self.train_model(train_loader, val_loader)
        
        # Step 5: Create visualizations
        self.create_training_visualizations()
        
        # Step 6: Save training report
        self.save_training_report()
        
        logger.info("Full training pipeline completed!")

class MultimodalDataset(torch.utils.data.Dataset):
    """Dataset class for multimodal data"""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.physio_data = torch.FloatTensor(data['physio'])
        self.facial_data = torch.FloatTensor(data['facial'])
        self.audio_data = torch.FloatTensor(data['audio'])
        self.labels = torch.FloatTensor(data['labels'])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.physio_data[idx],
            self.facial_data[idx],
            self.audio_data[idx],
            self.labels[idx]
        )

def main():
    """Main training script"""
    # Initialize training pipeline
    pipeline = MultimodalTrainingPipeline()
    
    # Run full training
    pipeline.run_full_training()
    
    print("Multimodal training completed successfully!")

if __name__ == "__main__":
    main() 