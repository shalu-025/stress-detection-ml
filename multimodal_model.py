"""
Multimodal stress detection model architecture
Combines physiological, facial, and audio modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class PhysiologicalEncoder(nn.Module):
    """Encoder for physiological data (WESAD)"""
    
    def __init__(self, input_dim: int = 25, hidden_dim: int = 128, latent_dim: int = 64):
        super(PhysiologicalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent representation
        self.latent_layer = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Output layer for stress prediction
        self.stress_predictor = nn.Linear(latent_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Latent representation
        latent = self.latent_layer(features)
        
        # Stress prediction
        stress_score = torch.sigmoid(self.stress_predictor(latent))
        
        return latent, stress_score

class FacialEncoder(nn.Module):
    """Encoder for facial data (ANUStressDB)"""
    
    def __init__(self, input_dim: int = 70, hidden_dim: int = 256, latent_dim: int = 64):
        super(FacialEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent representation
        self.latent_layer = nn.Linear(hidden_dim // 4, latent_dim)
        
        # Output layer for stress prediction
        self.stress_predictor = nn.Linear(latent_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Latent representation
        latent = self.latent_layer(features)
        
        # Stress prediction
        stress_score = torch.sigmoid(self.stress_predictor(latent))
        
        return latent, stress_score

class AudioEncoder(nn.Module):
    """Encoder for audio data (CREMA-D)"""
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 256, latent_dim: int = 64):
        super(AudioEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent representation
        self.latent_layer = nn.Linear(hidden_dim // 4, latent_dim)
        
        # Output layer for stress prediction
        self.stress_predictor = nn.Linear(latent_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Latent representation
        latent = self.latent_layer(features)
        
        # Stress prediction
        stress_score = torch.sigmoid(self.stress_predictor(latent))
        
        return latent, stress_score

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(self, latent_dim: int = 64, num_heads: int = 4):
        super(CrossModalAttention, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        
        # Multi-head attention
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention"""
        batch_size = x.size(0)
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        
        # Output projection
        output = self.output_proj(attended)
        
        return output

class MultimodalFusion(nn.Module):
    """Fusion module for combining multimodal representations"""
    
    def __init__(self, latent_dim: int = 64, fusion_dim: int = 128, num_modalities: int = 3):
        super(MultimodalFusion, self).__init__()
        
        self.latent_dim = latent_dim
        self.fusion_dim = fusion_dim
        self.num_modalities = num_modalities
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(latent_dim)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(latent_dim * num_modalities, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final stress prediction
        self.stress_predictor = nn.Linear(fusion_dim // 4, 1)
        
        # Uncertainty estimation
        self.uncertainty_predictor = nn.Linear(fusion_dim // 4, 1)
        
    def forward(self, modalities: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with multimodal fusion"""
        # Stack modalities
        stacked = torch.stack(modalities, dim=1)  # [batch, num_modalities, latent_dim]
        
        # Apply cross-modal attention
        attended = self.cross_attention(stacked)
        
        # Flatten for fusion
        flattened = attended.view(attended.size(0), -1)
        
        # Fusion
        fused = self.fusion_layers(flattened)
        
        # Stress prediction
        stress_score = torch.sigmoid(self.stress_predictor(fused))
        
        # Uncertainty estimation
        uncertainty = torch.sigmoid(self.uncertainty_predictor(fused))
        
        return stress_score, uncertainty, fused

class MultimodalStressModel(nn.Module):
    """Complete multimodal stress detection model"""
    
    def __init__(self, 
                 physio_input_dim: int = 25,
                 facial_input_dim: int = 70,
                 audio_input_dim: int = 40,
                 latent_dim: int = 64,
                 fusion_dim: int = 128):
        super(MultimodalStressModel, self).__init__()
        
        # Modality-specific encoders
        self.physio_encoder = PhysiologicalEncoder(physio_input_dim, 128, latent_dim)
        self.facial_encoder = FacialEncoder(facial_input_dim, 256, latent_dim)
        self.audio_encoder = AudioEncoder(audio_input_dim, 256, latent_dim)
        
        # Fusion module
        self.fusion = MultimodalFusion(latent_dim, fusion_dim)
        
        # Scalers for each modality
        self.physio_scaler = StandardScaler()
        self.facial_scaler = StandardScaler()
        self.audio_scaler = StandardScaler()
        
    def forward(self, physio_data: torch.Tensor, facial_data: torch.Tensor, 
                audio_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete model"""
        
        # Encode each modality
        physio_latent, physio_stress = self.physio_encoder(physio_data)
        facial_latent, facial_stress = self.facial_encoder(facial_data)
        audio_latent, audio_stress = self.audio_encoder(audio_data)
        
        # Multimodal fusion
        modalities = [physio_latent, facial_latent, audio_latent]
        fused_stress, uncertainty, fused_features = self.fusion(modalities)
        
        return {
            'physio_stress': physio_stress,
            'facial_stress': facial_stress,
            'audio_stress': audio_stress,
            'fused_stress': fused_stress,
            'uncertainty': uncertainty,
            'fused_features': fused_features,
            'physio_latent': physio_latent,
            'facial_latent': facial_latent,
            'audio_latent': audio_latent
        }
    
    def predict_stress_level(self, physio_data: np.ndarray, facial_data: np.ndarray, 
                           audio_data: np.ndarray) -> Dict[str, Any]:
        """Predict stress level with uncertainty"""
        self.eval()
        
        with torch.no_grad():
            # Convert to tensors
            physio_tensor = torch.FloatTensor(physio_data).unsqueeze(0)
            facial_tensor = torch.FloatTensor(facial_data).unsqueeze(0)
            audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
            
            # Forward pass
            outputs = self.forward(physio_tensor, facial_tensor, audio_tensor)
            
            # Extract predictions
            physio_score = outputs['physio_stress'].item()
            facial_score = outputs['facial_stress'].item()
            audio_score = outputs['audio_stress'].item()
            fused_score = outputs['fused_stress'].item()
            uncertainty = outputs['uncertainty'].item()
            
            # Determine stress level
            if fused_score < 0.3:
                stress_level = 'low'
            elif fused_score < 0.6:
                stress_level = 'medium'
            else:
                stress_level = 'high'
            
            return {
                'stress_level': stress_level,
                'stress_score': fused_score,
                'uncertainty': uncertainty,
                'physio_score': physio_score,
                'facial_score': facial_score,
                'audio_score': audio_score,
                'confidence': 1.0 - uncertainty
            }
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        try:
            model_data = {
                'model_state_dict': self.state_dict(),
                'physio_scaler': self.physio_scaler,
                'facial_scaler': self.facial_scaler,
                'audio_scaler': self.audio_scaler
            }
            torch.save(model_data, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load the complete model"""
        try:
            if os.path.exists(filepath):
                model_data = torch.load(filepath)
                self.load_state_dict(model_data['model_state_dict'])
                self.physio_scaler = model_data['physio_scaler']
                self.facial_scaler = model_data['facial_scaler']
                self.audio_scaler = model_data['audio_scaler']
                print(f"Model loaded from {filepath}")
            else:
                print(f"Model file not found: {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

class MultimodalTrainer:
    """Training class for the multimodal model"""
    
    def __init__(self, model: MultimodalStressModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss functions
        self.stress_loss = nn.BCELoss()
        self.uncertainty_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def train_step(self, physio_data: torch.Tensor, facial_data: torch.Tensor,
                  audio_data: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(physio_data, facial_data, audio_data)
        
        # Calculate losses
        stress_loss = self.stress_loss(outputs['fused_stress'], labels)
        
        # Total loss
        total_loss = stress_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'stress_loss': stress_loss.item()
        }
    
    def evaluate(self, physio_data: torch.Tensor, facial_data: torch.Tensor,
                audio_data: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(physio_data, facial_data, audio_data)
            
            # Calculate metrics
            stress_loss = self.stress_loss(outputs['fused_stress'], labels)
            
            # Accuracy
            predictions = (outputs['fused_stress'] > 0.5).float()
            accuracy = (predictions == labels).float().mean().item()
            
            return {
                'stress_loss': stress_loss.item(),
                'accuracy': accuracy
            }

def create_multimodal_model(physio_input_dim: int = 25, facial_input_dim: int = 70, 
                          audio_input_dim: int = 40) -> MultimodalStressModel:
    """Create a multimodal stress detection model"""
    return MultimodalStressModel(
        physio_input_dim=physio_input_dim,
        facial_input_dim=facial_input_dim,
        audio_input_dim=audio_input_dim,
        latent_dim=64,
        fusion_dim=128
    ) 