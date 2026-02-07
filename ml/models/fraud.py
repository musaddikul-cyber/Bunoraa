"""
Fraud Detection Models

Neural network models for detecting:
- Payment fraud
- Account takeover
- Suspicious order patterns
- Bot detection
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.base import BaseNeuralNetwork

logger = logging.getLogger("bunoraa.ml.fraud")


class AttentionAnomalyDetector(nn.Module):
    """
    Attention-based anomaly detection network.
    
    Uses self-attention to identify unusual patterns in transactions.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Transformer encoder for pattern detection
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Autoencoder for reconstruction error
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_features),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: (batch_size, num_features) or (batch_size, seq_len, num_features)
            return_attention: Whether to return attention weights
        
        Returns:
            Tuple of (fraud_score, reconstruction, attention_weights)
        """
        # Handle both single features and sequences
        if features.dim() == 2:
            features = features.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, _ = features.shape
        
        # Project input
        hidden = self.input_proj(features)
        
        # Encode with transformer
        encoded = self.encoder(hidden)
        
        # Pool sequence dimension
        pooled = encoded.mean(dim=1)
        
        # Reconstruction
        reconstruction = self.decoder(pooled)
        
        # Classification
        fraud_score = torch.sigmoid(self.classifier(pooled)).squeeze(-1)
        
        if return_attention:
            # Get attention from last layer (approximate)
            with torch.no_grad():
                attn = torch.softmax(
                    torch.matmul(encoded, encoded.transpose(-1, -2)) / np.sqrt(encoded.size(-1)),
                    dim=-1
                )
            return fraud_score, reconstruction, attn
        
        return fraud_score, reconstruction, None


class FraudDetectorNetwork(nn.Module):
    """
    Comprehensive fraud detection network.
    
    Combines multiple signals:
    - Transaction features
    - User behavior patterns
    - Device/location anomalies
    - Historical patterns
    """
    
    def __init__(
        self,
        num_transaction_features: int = 50,
        num_user_features: int = 30,
        num_device_features: int = 20,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Transaction encoder
        self.transaction_encoder = nn.Sequential(
            nn.Linear(num_transaction_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # User behavior encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(num_user_features, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        
        # Device/location encoder
        self.device_encoder = nn.Sequential(
            nn.Linear(num_device_features, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        
        # Cross-attention for feature interaction
        combined_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 4
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=combined_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Anomaly scorer
        self.anomaly_scorer = AttentionAnomalyDetector(
            num_features=combined_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Final classifier with multiple outputs
        self.fraud_classifier = nn.Sequential(
            nn.Linear(combined_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # 4 outputs: fraud_prob, chargeback_prob, bot_prob, takeover_prob
        )
        
        # Risk scoring head
        self.risk_scorer = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        transaction_features: torch.Tensor,
        user_features: torch.Tensor,
        device_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            transaction_features: (batch_size, num_transaction_features)
            user_features: (batch_size, num_user_features)
            device_features: (batch_size, num_device_features)
        
        Returns:
            Dict with fraud_prob, risk_score, and individual risk components
        """
        # Encode each feature set
        trans_encoded = self.transaction_encoder(transaction_features)
        user_encoded = self.user_encoder(user_features)
        device_encoded = self.device_encoder(device_features)
        
        # Combine features
        combined = torch.cat([trans_encoded, user_encoded, device_encoded], dim=-1)
        
        # Cross-attention for feature interaction
        combined_seq = combined.unsqueeze(1)  # Add sequence dim
        attn_output, _ = self.cross_attention(combined_seq, combined_seq, combined_seq)
        combined_attn = attn_output.squeeze(1)
        
        # Anomaly detection
        anomaly_score, reconstruction, _ = self.anomaly_scorer(combined)
        
        # Compute reconstruction error as additional signal
        recon_error = F.mse_loss(reconstruction, combined, reduction='none').mean(dim=-1)
        
        # Final classification
        final_features = torch.cat([combined_attn, recon_error.unsqueeze(-1).expand(-1, 128)], dim=-1)
        class_logits = self.fraud_classifier(final_features)
        
        # Risk score
        risk_score = torch.sigmoid(self.risk_scorer(combined_attn)).squeeze(-1)
        
        return {
            "fraud_prob": torch.sigmoid(class_logits[:, 0]),
            "chargeback_prob": torch.sigmoid(class_logits[:, 1]),
            "bot_prob": torch.sigmoid(class_logits[:, 2]),
            "takeover_prob": torch.sigmoid(class_logits[:, 3]),
            "risk_score": risk_score,
            "anomaly_score": anomaly_score,
            "reconstruction_error": recon_error,
        }


class FraudDetector(BaseNeuralNetwork):
    """
    Fraud detection model for e-commerce transactions.
    """
    
    MODEL_TYPE = "fraud_detection"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "fraud_detector",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_transaction_features": 50,
            "num_user_features": 30,
            "num_device_features": 20,
            "hidden_dim": 128,
            "num_attention_heads": 4,
            "dropout": 0.2,
            "learning_rate": 0.0005,
            "batch_size": 256,
            "epochs": 30,
            "fraud_weight": 10.0,  # Weight for fraud class (imbalanced)
            "threshold": 0.5,  # Classification threshold
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return FraudDetectorNetwork(
            num_transaction_features=self.config["num_transaction_features"],
            num_user_features=self.config["num_user_features"],
            num_device_features=self.config["num_device_features"],
            hidden_dim=self.config["hidden_dim"],
            num_attention_heads=self.config["num_attention_heads"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Weighted BCE for imbalanced fraud detection."""
        fraud_weight = self.config["fraud_weight"]
        
        class FraudLoss(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
            
            def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
                # Main fraud loss with class weighting
                fraud_prob = outputs["fraud_prob"]
                weight = torch.where(targets == 1, self.weight, 1.0)
                bce = F.binary_cross_entropy(fraud_prob, targets, weight=weight)
                
                # Reconstruction loss for anomaly detection
                recon_loss = outputs["reconstruction_error"].mean()
                
                return bce + 0.1 * recon_loss
        
        return FraudLoss(fraud_weight)
    
    def detect_fraud(
        self,
        transaction_features: np.ndarray,
        user_features: np.ndarray,
        device_features: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect fraud for a transaction.
        
        Args:
            transaction_features: Transaction feature vector
            user_features: User behavior features
            device_features: Device/location features
            threshold: Classification threshold (default from config)
        
        Returns:
            Dict with predictions and risk assessment
        """
        self.model.eval()
        threshold = threshold or self.config["threshold"]
        
        with torch.no_grad():
            trans = torch.FloatTensor([transaction_features]).to(self.device)
            user = torch.FloatTensor([user_features]).to(self.device)
            device = torch.FloatTensor([device_features]).to(self.device)
            
            outputs = self.model(trans, user, device)
        
        result = {
            "is_fraud": outputs["fraud_prob"].item() > threshold,
            "fraud_probability": outputs["fraud_prob"].item(),
            "risk_score": outputs["risk_score"].item(),
            "risk_level": self._get_risk_level(outputs["risk_score"].item()),
            "flags": [],
        }
        
        # Check individual risk factors
        if outputs["chargeback_prob"].item() > 0.5:
            result["flags"].append("high_chargeback_risk")
        if outputs["bot_prob"].item() > 0.5:
            result["flags"].append("bot_suspected")
        if outputs["takeover_prob"].item() > 0.5:
            result["flags"].append("account_takeover_risk")
        if outputs["anomaly_score"].item() > 0.7:
            result["flags"].append("anomalous_behavior")
        
        # Detailed scores
        result["scores"] = {
            "fraud": outputs["fraud_prob"].item(),
            "chargeback": outputs["chargeback_prob"].item(),
            "bot": outputs["bot_prob"].item(),
            "takeover": outputs["takeover_prob"].item(),
            "anomaly": outputs["anomaly_score"].item(),
        }
        
        return result
    
    def _get_risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def batch_detect(
        self,
        transactions: List[Dict[str, np.ndarray]],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch fraud detection.
        
        Args:
            transactions: List of dicts with transaction/user/device features
            threshold: Classification threshold
        
        Returns:
            List of detection results
        """
        results = []
        for tx in transactions:
            result = self.detect_fraud(
                tx["transaction_features"],
                tx["user_features"],
                tx["device_features"],
                threshold
            )
            results.append(result)
        return results
    
    def get_feature_importance(
        self,
        sample_transactions: List[Dict[str, np.ndarray]],
        feature_names: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute feature importance using gradient-based attribution.
        
        Args:
            sample_transactions: Sample transactions for analysis
            feature_names: Optional dict mapping feature type to names
        
        Returns:
            Dict of feature importances by category
        """
        self.model.eval()
        importances = {
            "transaction": {},
            "user": {},
            "device": {},
        }
        
        # Use integrated gradients approximation
        for tx in sample_transactions:
            trans = torch.FloatTensor([tx["transaction_features"]]).to(self.device)
            user = torch.FloatTensor([tx["user_features"]]).to(self.device)
            device = torch.FloatTensor([tx["device_features"]]).to(self.device)
            
            trans.requires_grad = True
            user.requires_grad = True
            device.requires_grad = True
            
            outputs = self.model(trans, user, device)
            outputs["fraud_prob"].backward()
            
            # Aggregate gradients
            for i, grad in enumerate(trans.grad[0].cpu().numpy()):
                key = f"trans_{i}" if not feature_names else feature_names.get("transaction", [])[i] if i < len(feature_names.get("transaction", [])) else f"trans_{i}"
                importances["transaction"][key] = importances["transaction"].get(key, 0) + abs(grad)
            
            for i, grad in enumerate(user.grad[0].cpu().numpy()):
                key = f"user_{i}" if not feature_names else feature_names.get("user", [])[i] if i < len(feature_names.get("user", [])) else f"user_{i}"
                importances["user"][key] = importances["user"].get(key, 0) + abs(grad)
            
            for i, grad in enumerate(device.grad[0].cpu().numpy()):
                key = f"device_{i}" if not feature_names else feature_names.get("device", [])[i] if i < len(feature_names.get("device", [])) else f"device_{i}"
                importances["device"][key] = importances["device"].get(key, 0) + abs(grad)
        
        # Normalize
        n = len(sample_transactions)
        for category in importances:
            for key in importances[category]:
                importances[category][key] /= n
        
        return importances
