"""
Forecasting Models for Demand and Price Optimization

Models for:
- Demand forecasting (time series)
- Price elasticity prediction
- Dynamic pricing optimization
- Inventory optimization
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.base import BaseNeuralNetwork, BaseMLModel

logger = logging.getLogger("bunoraa.ml.forecasting")


# ==================== Demand Forecasting ====================

class TemporalFusionBlock(nn.Module):
    """
    Temporal Fusion Block for time series modeling.
    Combines LSTM with attention for interpretable forecasts.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # GRN (Gated Residual Network)
        self.grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # GRN with gating
        residual = x if x.size(-1) == self.grn[0].out_features else \
                   F.pad(x, (0, self.grn[0].out_features - x.size(-1)))
        hidden = self.grn(x)
        gated = self.gate(hidden) * hidden
        output = self.layer_norm(gated + residual)
        
        # Self-attention
        attn_output, _ = self.attention(output, output, output, key_padding_mask=mask)
        output = self.attention_norm(output + attn_output)
        
        return output


class DemandForecasterNetwork(nn.Module):
    """
    Neural network for demand forecasting.
    
    Features:
    - Multi-horizon forecasting
    - Interpretable attention
    - Covariate handling (promotions, holidays, etc.)
    - Quantile regression for uncertainty
    """
    
    def __init__(
        self,
        num_products: int,
        num_stores: int = 1,
        num_static_features: int = 10,
        num_temporal_features: int = 20,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        forecast_horizon: int = 14,
        num_quantiles: int = 3,  # [0.1, 0.5, 0.9]
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.forecast_horizon = forecast_horizon
        self.num_quantiles = num_quantiles
        
        # Embeddings
        self.product_embedding = nn.Embedding(num_products, 32)
        self.store_embedding = nn.Embedding(num_stores, 16)
        
        # Static encoder
        static_input_dim = 32 + 16 + num_static_features
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Temporal encoder (LSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=num_temporal_features,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Temporal fusion
        self.temporal_fusion = TemporalFusionBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim + num_temporal_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output projections (one per quantile)
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_quantiles)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        product_ids: torch.Tensor,
        store_ids: torch.Tensor,
        static_features: torch.Tensor,
        historical_sequence: torch.Tensor,
        future_covariates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            product_ids: (batch_size,) product indices
            store_ids: (batch_size,) store indices
            static_features: (batch_size, num_static_features)
            historical_sequence: (batch_size, history_length, num_temporal_features)
            future_covariates: (batch_size, forecast_horizon, num_temporal_features)
        
        Returns:
            (batch_size, forecast_horizon, num_quantiles) demand predictions
        """
        batch_size = product_ids.size(0)
        
        # Static encoding
        prod_emb = self.product_embedding(product_ids)
        store_emb = self.store_embedding(store_ids)
        static_combined = torch.cat([prod_emb, store_emb, static_features], dim=-1)
        static_encoded = self.static_encoder(static_combined)
        
        # Temporal encoding
        temporal_output, (h_n, c_n) = self.temporal_encoder(historical_sequence)
        temporal_fused = self.temporal_fusion(temporal_output)
        
        # Get context from last timestep
        context = temporal_fused[:, -1, :]  # (batch_size, hidden_dim)
        
        # Initialize decoder with context + static
        decoder_hidden = (
            (context + static_encoded).unsqueeze(0),
            torch.zeros_like(context).unsqueeze(0)
        )
        
        # Decode future
        decoder_input = future_covariates
        decoder_context = context.unsqueeze(1).expand(-1, self.forecast_horizon, -1)
        decoder_combined = torch.cat([decoder_context, decoder_input], dim=-1)
        
        decoder_output, _ = self.decoder(decoder_combined, decoder_hidden)
        
        # Generate quantile predictions
        quantile_outputs = []
        for output_layer in self.output_layers:
            quantile_pred = output_layer(decoder_output)
            quantile_outputs.append(quantile_pred)
        
        # Stack: (batch_size, forecast_horizon, num_quantiles)
        predictions = torch.cat(quantile_outputs, dim=-1)
        
        # Ensure quantiles are ordered
        predictions = torch.sort(predictions, dim=-1)[0]
        
        return F.softplus(predictions)  # Ensure positive demand


class DemandForecaster(BaseNeuralNetwork):
    """
    Demand forecasting model.
    """
    
    MODEL_TYPE = "forecasting"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "demand_forecaster",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_products": 50000,
            "num_stores": 100,
            "num_static_features": 10,
            "num_temporal_features": 20,
            "hidden_dim": 128,
            "num_lstm_layers": 2,
            "forecast_horizon": 14,
            "num_quantiles": 3,
            "quantiles": [0.1, 0.5, 0.9],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return DemandForecasterNetwork(
            num_products=self.config["num_products"],
            num_stores=self.config["num_stores"],
            num_static_features=self.config["num_static_features"],
            num_temporal_features=self.config["num_temporal_features"],
            hidden_dim=self.config["hidden_dim"],
            num_lstm_layers=self.config["num_lstm_layers"],
            forecast_horizon=self.config["forecast_horizon"],
            num_quantiles=self.config["num_quantiles"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Quantile loss for probabilistic forecasting."""
        quantiles = self.config.get("quantiles", [0.1, 0.5, 0.9])
        
        class QuantileLoss(nn.Module):
            def __init__(self, quantiles):
                super().__init__()
                self.quantiles = quantiles
            
            def forward(self, predictions, targets):
                # predictions: (batch, horizon, num_quantiles)
                # targets: (batch, horizon)
                losses = []
                for i, q in enumerate(self.quantiles):
                    pred = predictions[:, :, i]
                    error = targets - pred
                    loss = torch.max(q * error, (q - 1) * error)
                    losses.append(loss.mean())
                return sum(losses) / len(losses)
        
        return QuantileLoss(quantiles)
    
    def forecast(
        self,
        product_id: int,
        store_id: int,
        static_features: np.ndarray,
        history: np.ndarray,
        future_covariates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate demand forecast.
        
        Returns:
            Dict with 'low', 'median', 'high' forecasts
        """
        self.model.eval()
        
        with torch.no_grad():
            product_ids = torch.LongTensor([product_id]).to(self.device)
            store_ids = torch.LongTensor([store_id]).to(self.device)
            static = torch.FloatTensor([static_features]).to(self.device)
            hist = torch.FloatTensor([history]).to(self.device)
            future = torch.FloatTensor([future_covariates]).to(self.device)
            
            predictions = self.model(product_ids, store_ids, static, hist, future)
            predictions = predictions.cpu().numpy()[0]
        
        quantile_names = ['low', 'median', 'high']
        return {
            name: predictions[:, i]
            for i, name in enumerate(quantile_names[:self.config["num_quantiles"]])
        }


# ==================== Price Optimization ====================

class PriceOptimizerNetwork(nn.Module):
    """
    Neural network for price optimization.
    
    Learns price elasticity and optimal pricing strategies.
    """
    
    def __init__(
        self,
        num_products: int,
        num_categories: int,
        num_features: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [256, 128, 64]
        
        # Embeddings
        self.product_embedding = nn.Embedding(num_products, 64)
        self.category_embedding = nn.Embedding(num_categories, 32)
        
        # Feature encoder
        input_dim = 64 + 32 + num_features + 1  # +1 for price input
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads
        self.demand_head = nn.Linear(prev_dim, 1)  # Predicted demand
        self.elasticity_head = nn.Linear(prev_dim, 1)  # Price elasticity
        self.optimal_price_head = nn.Linear(prev_dim, 1)  # Suggested price
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        product_ids: torch.Tensor,
        category_ids: torch.Tensor,
        features: torch.Tensor,
        prices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            product_ids: (batch_size,) product indices
            category_ids: (batch_size,) category indices
            features: (batch_size, num_features) product/market features
            prices: (batch_size, 1) current prices
        
        Returns:
            Tuple of (predicted_demand, elasticity, optimal_price)
        """
        prod_emb = self.product_embedding(product_ids)
        cat_emb = self.category_embedding(category_ids)
        
        combined = torch.cat([prod_emb, cat_emb, features, prices], dim=-1)
        encoded = self.encoder(combined)
        
        demand = F.softplus(self.demand_head(encoded))
        elasticity = self.elasticity_head(encoded)  # Can be negative
        optimal_price = F.softplus(self.optimal_price_head(encoded))
        
        return demand.squeeze(-1), elasticity.squeeze(-1), optimal_price.squeeze(-1)


class PriceOptimizer(BaseNeuralNetwork):
    """
    Price optimization model.
    """
    
    MODEL_TYPE = "optimization"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "price_optimizer",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_products": 50000,
            "num_categories": 500,
            "num_features": 30,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 30,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return PriceOptimizerNetwork(
            num_products=self.config["num_products"],
            num_categories=self.config["num_categories"],
            num_features=self.config["num_features"],
            hidden_dims=self.config["hidden_dims"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Combined loss for demand prediction and profit optimization."""
        class PricingLoss(nn.Module):
            def forward(self, pred_demand, true_demand, pred_price, cost, elasticity):
                # Demand MSE
                demand_loss = F.mse_loss(pred_demand, true_demand)
                
                # Revenue = price * demand
                # We want to maximize: (price - cost) * demand
                # As loss, we minimize negative profit
                profit = (pred_price - cost) * pred_demand
                profit_loss = -profit.mean()
                
                return demand_loss + 0.1 * profit_loss
        
        return PricingLoss()
    
    def get_optimal_price(
        self,
        product_id: int,
        category_id: int,
        features: np.ndarray,
        cost: float,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        margin_target: float = 0.3
    ) -> Dict[str, float]:
        """
        Get optimal price for a product.
        
        Args:
            product_id: Product ID
            category_id: Category ID
            features: Product/market features
            cost: Product cost
            min_price: Minimum allowed price
            max_price: Maximum allowed price
            margin_target: Target profit margin
        
        Returns:
            Dict with optimal_price, predicted_demand, elasticity, expected_profit
        """
        self.model.eval()
        
        # Try range of prices
        if min_price is None:
            min_price = cost * (1 + margin_target * 0.5)
        if max_price is None:
            max_price = cost * (1 + margin_target * 3)
        
        test_prices = np.linspace(min_price, max_price, 50)
        best_profit = -np.inf
        best_result = None
        
        with torch.no_grad():
            for price in test_prices:
                product_ids = torch.LongTensor([product_id]).to(self.device)
                category_ids = torch.LongTensor([category_id]).to(self.device)
                feats = torch.FloatTensor([features]).to(self.device)
                prices = torch.FloatTensor([[price]]).to(self.device)
                
                demand, elasticity, _ = self.model(product_ids, category_ids, feats, prices)
                
                demand = demand.item()
                elasticity = elasticity.item()
                profit = (price - cost) * demand
                
                if profit > best_profit:
                    best_profit = profit
                    best_result = {
                        "optimal_price": price,
                        "predicted_demand": demand,
                        "elasticity": elasticity,
                        "expected_profit": profit,
                        "margin": (price - cost) / price
                    }
        
        return best_result
    
    def simulate_price_change(
        self,
        product_id: int,
        category_id: int,
        features: np.ndarray,
        current_price: float,
        new_price: float
    ) -> Dict[str, float]:
        """Simulate the impact of a price change."""
        self.model.eval()
        
        with torch.no_grad():
            product_ids = torch.LongTensor([product_id]).to(self.device)
            category_ids = torch.LongTensor([category_id]).to(self.device)
            feats = torch.FloatTensor([features]).to(self.device)
            
            # Current price demand
            current_prices = torch.FloatTensor([[current_price]]).to(self.device)
            current_demand, _, _ = self.model(product_ids, category_ids, feats, current_prices)
            
            # New price demand
            new_prices = torch.FloatTensor([[new_price]]).to(self.device)
            new_demand, elasticity, _ = self.model(product_ids, category_ids, feats, new_prices)
        
        price_change_pct = (new_price - current_price) / current_price * 100
        demand_change_pct = (new_demand.item() - current_demand.item()) / current_demand.item() * 100
        
        return {
            "price_change_pct": price_change_pct,
            "demand_change_pct": demand_change_pct,
            "elasticity": elasticity.item(),
            "current_demand": current_demand.item(),
            "new_demand": new_demand.item(),
        }
