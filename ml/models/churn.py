"""
Customer Churn Prediction and Lifetime Value Models

Models for predicting:
- Customer churn probability
- Customer lifetime value (CLV)
- Next purchase timing
- Customer engagement scoring
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

logger = logging.getLogger("bunoraa.ml.churn")


class ChurnPredictorNetwork(nn.Module):
    """
    Neural network for predicting customer churn.
    
    Multi-task learning approach:
    - Churn prediction (classification)
    - Time to churn (regression)
    - Churn reason classification
    """
    
    def __init__(
        self,
        num_behavioral_features: int = 50,
        num_transaction_features: int = 30,
        num_demographic_features: int = 20,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        max_history_length: int = 50,
        num_churn_reasons: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Behavioral pattern encoder (for time series of user actions)
        self.behavioral_encoder = nn.GRU(
            input_size=num_behavioral_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention for behavioral patterns
        self.behavioral_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Transaction encoder
        self.transaction_encoder = nn.Sequential(
            nn.Linear(num_transaction_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Demographic encoder
        self.demographic_encoder = nn.Sequential(
            nn.Linear(num_demographic_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
        )
        
        # Feature fusion
        fusion_dim = hidden_dim * 2 + hidden_dim // 2 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multi-task heads
        # 1. Churn probability
        self.churn_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # 2. Days until churn (survival analysis style)
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # 3. Churn reason classification
        self.reason_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_churn_reasons),
        )
        
        # 4. Risk factors extraction
        self.risk_factors_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 10),  # Top 10 risk factor weights
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
        behavioral_history: torch.Tensor,
        behavioral_mask: Optional[torch.Tensor],
        transaction_features: torch.Tensor,
        demographic_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            behavioral_history: (batch_size, seq_len, num_behavioral_features)
            behavioral_mask: (batch_size, seq_len) mask for padding
            transaction_features: (batch_size, num_transaction_features)
            demographic_features: (batch_size, num_demographic_features)
        
        Returns:
            Dict with churn predictions
        """
        # Encode behavioral history
        behavioral_output, _ = self.behavioral_encoder(behavioral_history)
        
        # Attention pooling
        attn_weights = self.behavioral_attention(behavioral_output)
        if behavioral_mask is not None:
            attn_weights = attn_weights.masked_fill(
                ~behavioral_mask.unsqueeze(-1).bool(),
                float('-inf')
            )
        attn_weights = F.softmax(attn_weights, dim=1)
        behavioral_encoded = (behavioral_output * attn_weights).sum(dim=1)
        
        # Encode other features
        transaction_encoded = self.transaction_encoder(transaction_features)
        demographic_encoded = self.demographic_encoder(demographic_features)
        
        # Fuse all features
        combined = torch.cat([
            behavioral_encoded,
            transaction_encoded,
            demographic_encoded
        ], dim=-1)
        
        fused = self.fusion(combined)
        
        # Multi-task outputs
        churn_prob = torch.sigmoid(self.churn_head(fused)).squeeze(-1)
        days_to_churn = F.softplus(self.time_head(fused)).squeeze(-1)  # Positive
        churn_reason_logits = self.reason_head(fused)
        risk_factors = torch.sigmoid(self.risk_factors_head(fused))
        
        return {
            "churn_prob": churn_prob,
            "days_to_churn": days_to_churn,
            "churn_reason_logits": churn_reason_logits,
            "risk_factors": risk_factors,
            "attention_weights": attn_weights.squeeze(-1),
        }


class ChurnPredictor(BaseNeuralNetwork):
    """
    Customer churn prediction model.
    """
    
    MODEL_TYPE = "churn_prediction"
    FRAMEWORK = "pytorch"
    
    CHURN_REASONS = [
        "price_sensitivity",
        "poor_experience",
        "competitor",
        "product_issues",
        "natural_lifecycle"
    ]
    
    def __init__(
        self,
        model_name: str = "churn_predictor",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_behavioral_features": 50,
            "num_transaction_features": 30,
            "num_demographic_features": 20,
            "hidden_dim": 128,
            "num_attention_heads": 4,
            "max_history_length": 50,
            "num_churn_reasons": 5,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 30,
            "churn_weight": 3.0,  # Class weight for imbalanced data
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return ChurnPredictorNetwork(
            num_behavioral_features=self.config["num_behavioral_features"],
            num_transaction_features=self.config["num_transaction_features"],
            num_demographic_features=self.config["num_demographic_features"],
            hidden_dim=self.config["hidden_dim"],
            num_attention_heads=self.config["num_attention_heads"],
            max_history_length=self.config["max_history_length"],
            num_churn_reasons=self.config["num_churn_reasons"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Multi-task loss for churn prediction."""
        churn_weight = self.config["churn_weight"]
        
        class ChurnLoss(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
            
            def forward(
                self,
                outputs: Dict[str, torch.Tensor],
                churn_labels: torch.Tensor,
                days_labels: Optional[torch.Tensor] = None,
                reason_labels: Optional[torch.Tensor] = None
            ):
                # Churn prediction loss (weighted BCE)
                weight = torch.where(churn_labels == 1, self.weight, 1.0)
                churn_loss = F.binary_cross_entropy(
                    outputs["churn_prob"],
                    churn_labels,
                    weight=weight
                )
                
                total_loss = churn_loss
                
                # Days to churn loss (only for churned users)
                if days_labels is not None:
                    churned_mask = churn_labels == 1
                    if churned_mask.any():
                        days_loss = F.mse_loss(
                            outputs["days_to_churn"][churned_mask],
                            days_labels[churned_mask]
                        )
                        total_loss = total_loss + 0.1 * days_loss
                
                # Reason classification loss
                if reason_labels is not None:
                    churned_mask = churn_labels == 1
                    if churned_mask.any():
                        reason_loss = F.cross_entropy(
                            outputs["churn_reason_logits"][churned_mask],
                            reason_labels[churned_mask]
                        )
                        total_loss = total_loss + 0.1 * reason_loss
                
                return total_loss
        
        return ChurnLoss(churn_weight)
    
    def predict_churn(
        self,
        behavioral_history: np.ndarray,
        transaction_features: np.ndarray,
        demographic_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predict churn for a customer.
        
        Args:
            behavioral_history: (seq_len, num_behavioral_features) history
            transaction_features: (num_transaction_features,) features
            demographic_features: (num_demographic_features,) features
        
        Returns:
            Dict with churn prediction and insights
        """
        self.model.eval()
        max_len = self.config["max_history_length"]
        
        # Prepare behavioral history
        if len(behavioral_history) > max_len:
            behavioral_history = behavioral_history[-max_len:]
        
        # Pad if needed
        seq_len = len(behavioral_history)
        if seq_len < max_len:
            padding = np.zeros((max_len - seq_len, behavioral_history.shape[-1]))
            behavioral_history = np.vstack([padding, behavioral_history])
            mask = [0] * (max_len - seq_len) + [1] * seq_len
        else:
            mask = [1] * max_len
        
        with torch.no_grad():
            history = torch.FloatTensor([behavioral_history]).to(self.device)
            mask_tensor = torch.FloatTensor([mask]).to(self.device)
            trans = torch.FloatTensor([transaction_features]).to(self.device)
            demo = torch.FloatTensor([demographic_features]).to(self.device)
            
            outputs = self.model(history, mask_tensor, trans, demo)
        
        churn_prob = outputs["churn_prob"].item()
        
        result = {
            "churn_probability": churn_prob,
            "risk_level": self._get_risk_level(churn_prob),
            "days_until_churn": int(outputs["days_to_churn"].item()) if churn_prob > 0.5 else None,
            "primary_churn_reason": None,
            "risk_factors": {},
            "recommendations": [],
        }
        
        # Get churn reason if high probability
        if churn_prob > 0.3:
            reason_probs = F.softmax(outputs["churn_reason_logits"], dim=-1)[0].cpu().numpy()
            reason_idx = int(np.argmax(reason_probs))
            result["primary_churn_reason"] = {
                "reason": self.CHURN_REASONS[reason_idx],
                "confidence": float(reason_probs[reason_idx])
            }
            result["all_reason_probabilities"] = {
                reason: float(prob)
                for reason, prob in zip(self.CHURN_REASONS, reason_probs)
            }
        
        # Risk factors
        risk_weights = outputs["risk_factors"][0].cpu().numpy()
        risk_factor_names = [
            "purchase_frequency_decline",
            "engagement_drop",
            "support_tickets",
            "price_sensitivity",
            "competitor_browsing",
            "negative_feedback",
            "payment_issues",
            "delivery_complaints",
            "inactive_period",
            "cart_abandonment"
        ]
        result["risk_factors"] = {
            name: float(weight)
            for name, weight in zip(risk_factor_names, risk_weights)
        }
        
        # Generate recommendations
        result["recommendations"] = self._generate_recommendations(result)
        
        return result
    
    def _get_risk_level(self, prob: float) -> str:
        if prob >= 0.8:
            return "critical"
        elif prob >= 0.6:
            return "high"
        elif prob >= 0.4:
            return "medium"
        elif prob >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        if result["risk_level"] in ["critical", "high"]:
            recommendations.append("Initiate proactive outreach with personalized offer")
            
            reason = result.get("primary_churn_reason", {}).get("reason", "")
            if reason == "price_sensitivity":
                recommendations.append("Consider loyalty discount or bundle offer")
            elif reason == "poor_experience":
                recommendations.append("Escalate to customer success team")
            elif reason == "competitor":
                recommendations.append("Highlight unique value propositions")
            elif reason == "product_issues":
                recommendations.append("Offer product exchange or credit")
        
        risk_factors = result.get("risk_factors", {})
        if risk_factors.get("engagement_drop", 0) > 0.7:
            recommendations.append("Send re-engagement campaign")
        if risk_factors.get("cart_abandonment", 0) > 0.7:
            recommendations.append("Send cart recovery email with incentive")
        
        return recommendations


# ==================== Customer Lifetime Value ====================

class CLVNetwork(nn.Module):
    """
    Customer Lifetime Value prediction network.
    
    Predicts:
    - Total future value
    - Expected transaction frequency
    - Average order value
    - Customer lifespan
    """
    
    def __init__(
        self,
        num_features: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Feature encoder
        layers = []
        prev_dim = num_features
        for i in range(num_layers):
            hidden = hidden_dim // (2 ** i) if i > 0 else hidden_dim
            layers.extend([
                nn.Linear(prev_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads
        self.clv_head = nn.Linear(prev_dim, 1)  # Total lifetime value
        self.frequency_head = nn.Linear(prev_dim, 1)  # Expected purchases per year
        self.aov_head = nn.Linear(prev_dim, 1)  # Average order value
        self.lifespan_head = nn.Linear(prev_dim, 1)  # Expected months active
        
        # Uncertainty head (for probabilistic predictions)
        self.uncertainty_head = nn.Linear(prev_dim, 4)  # Variance for each prediction
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: (batch_size, num_features) customer features
        
        Returns:
            Dict with CLV predictions
        """
        encoded = self.encoder(features)
        
        # Predictions (all positive via softplus)
        clv = F.softplus(self.clv_head(encoded)).squeeze(-1)
        frequency = F.softplus(self.frequency_head(encoded)).squeeze(-1)
        aov = F.softplus(self.aov_head(encoded)).squeeze(-1)
        lifespan = F.softplus(self.lifespan_head(encoded)).squeeze(-1)
        
        # Uncertainty (variance)
        uncertainty = F.softplus(self.uncertainty_head(encoded))
        
        return {
            "clv": clv,
            "frequency": frequency,
            "aov": aov,
            "lifespan_months": lifespan,
            "clv_uncertainty": uncertainty[:, 0],
            "frequency_uncertainty": uncertainty[:, 1],
            "aov_uncertainty": uncertainty[:, 2],
            "lifespan_uncertainty": uncertainty[:, 3],
        }


class CustomerLifetimeValue(BaseNeuralNetwork):
    """
    Customer Lifetime Value prediction model.
    """
    
    MODEL_TYPE = "clv_prediction"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "clv_predictor",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_features": 100,
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 50,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return CLVNetwork(
            num_features=self.config["num_features"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Negative log-likelihood loss with uncertainty."""
        class CLVLoss(nn.Module):
            def forward(
                self,
                outputs: Dict[str, torch.Tensor],
                clv_labels: torch.Tensor,
                frequency_labels: Optional[torch.Tensor] = None,
                aov_labels: Optional[torch.Tensor] = None
            ):
                # Gaussian NLL for CLV
                clv_loss = F.gaussian_nll_loss(
                    outputs["clv"],
                    clv_labels,
                    outputs["clv_uncertainty"]
                )
                
                total_loss = clv_loss
                
                if frequency_labels is not None:
                    freq_loss = F.gaussian_nll_loss(
                        outputs["frequency"],
                        frequency_labels,
                        outputs["frequency_uncertainty"]
                    )
                    total_loss = total_loss + 0.3 * freq_loss
                
                if aov_labels is not None:
                    aov_loss = F.gaussian_nll_loss(
                        outputs["aov"],
                        aov_labels,
                        outputs["aov_uncertainty"]
                    )
                    total_loss = total_loss + 0.3 * aov_loss
                
                return total_loss
        
        return CLVLoss()
    
    def predict_clv(
        self,
        customer_features: np.ndarray,
        include_breakdown: bool = True
    ) -> Dict[str, Any]:
        """
        Predict customer lifetime value.
        
        Args:
            customer_features: (num_features,) customer feature vector
            include_breakdown: Whether to include CLV breakdown
        
        Returns:
            Dict with CLV prediction and breakdown
        """
        self.model.eval()
        
        with torch.no_grad():
            features = torch.FloatTensor([customer_features]).to(self.device)
            outputs = self.model(features)
        
        clv = outputs["clv"].item()
        clv_std = np.sqrt(outputs["clv_uncertainty"].item())
        
        result = {
            "clv": clv,
            "clv_range": {
                "low": max(0, clv - 2 * clv_std),
                "high": clv + 2 * clv_std
            },
            "confidence_interval": 0.95,
            "segment": self._get_value_segment(clv),
        }
        
        if include_breakdown:
            result["breakdown"] = {
                "expected_annual_purchases": outputs["frequency"].item(),
                "average_order_value": outputs["aov"].item(),
                "expected_lifespan_months": outputs["lifespan_months"].item(),
            }
            
            # Validate CLV ~= frequency * AOV * lifespan/12
            computed_clv = (
                result["breakdown"]["expected_annual_purchases"] *
                result["breakdown"]["average_order_value"] *
                result["breakdown"]["expected_lifespan_months"] / 12
            )
            result["breakdown"]["computed_clv"] = computed_clv
        
        return result
    
    def _get_value_segment(self, clv: float) -> str:
        if clv >= 50000:
            return "platinum"
        elif clv >= 20000:
            return "gold"
        elif clv >= 5000:
            return "silver"
        elif clv >= 1000:
            return "bronze"
        else:
            return "standard"
    
    def segment_customers(
        self,
        customer_features: np.ndarray,
        customer_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Segment customers by CLV.
        
        Args:
            customer_features: (num_customers, num_features) features
            customer_ids: List of customer IDs
        
        Returns:
            Dict mapping segments to customer lists
        """
        segments = {
            "platinum": [],
            "gold": [],
            "silver": [],
            "bronze": [],
            "standard": [],
        }
        
        self.model.eval()
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(customer_features), batch_size):
                batch = torch.FloatTensor(customer_features[i:i + batch_size]).to(self.device)
                outputs = self.model(batch)
                
                for j, clv in enumerate(outputs["clv"].cpu().numpy()):
                    customer_idx = i + j
                    segment = self._get_value_segment(clv)
                    segments[segment].append({
                        "customer_id": customer_ids[customer_idx],
                        "clv": float(clv)
                    })
        
        # Sort each segment by CLV
        for segment in segments:
            segments[segment].sort(key=lambda x: x["clv"], reverse=True)
        
        return segments
