"""
Deep Learning Recommendation Models

State-of-the-art recommendation systems:
- Neural Collaborative Filtering (NCF)
- DeepFM (Deep Factorization Machine)
- Two-Tower Model (for retrieval)
- Sequential Recommender (BERT4Rec style)
- Multi-Interest Network
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

logger = logging.getLogger("bunoraa.ml.recommender")


# ==================== Neural Collaborative Filtering ====================

class NCFNetwork(nn.Module):
    """
    Neural Collaborative Filtering combining GMF and MLP.
    
    Reference: He et al. "Neural Collaborative Filtering" (WWW 2017)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_hidden_dims: List[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        mlp_hidden_dims = mlp_hidden_dims or [128, 64, 32]
        
        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings (separate from GMF)
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_input_dim = embedding_dim * 2
        layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        
        # NeuMF (combination layer)
        neumf_input_dim = embedding_dim + mlp_hidden_dims[-1]
        self.neumf = nn.Linear(neumf_input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: (batch_size,) user indices
            item_ids: (batch_size,) item indices
        
        Returns:
            (batch_size,) prediction scores
        """
        # GMF path
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item  # Element-wise product
        
        # MLP path
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.neumf(combined).squeeze(-1)
        
        return torch.sigmoid(output)


class NeuralCollaborativeFiltering(BaseNeuralNetwork):
    """
    Neural Collaborative Filtering model for user-item prediction.
    """
    
    MODEL_TYPE = "recommendation"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "ncf",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_users": 100000,
            "num_items": 50000,
            "embedding_dim": 64,
            "mlp_hidden_dims": [128, 64, 32],
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 1024,
            "epochs": 20,
            "negative_samples": 4,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return NCFNetwork(
            num_users=self.config["num_users"],
            num_items=self.config["num_items"],
            embedding_dim=self.config["embedding_dim"],
            mlp_hidden_dims=self.config["mlp_hidden_dims"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        return nn.BCELoss()
    
    def predict_scores(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        batch_size: int = 1024
    ) -> np.ndarray:
        """Predict interaction scores for user-item pairs."""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(user_ids), batch_size):
                users = torch.LongTensor(user_ids[i:i + batch_size]).to(self.device)
                items = torch.LongTensor(item_ids[i:i + batch_size]).to(self.device)
                batch_scores = self.model(users, items)
                scores.append(batch_scores.cpu().numpy())
        
        return np.concatenate(scores)
    
    def recommend_for_user(
        self,
        user_id: int,
        candidate_items: np.ndarray,
        top_k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Get top-K recommendations for a user.
        
        Args:
            user_id: User ID
            candidate_items: Array of candidate item IDs
            top_k: Number of recommendations
            exclude_items: Items to exclude
        
        Returns:
            List of (item_id, score) tuples
        """
        if exclude_items:
            candidate_items = np.array([i for i in candidate_items if i not in exclude_items])
        
        user_ids = np.full(len(candidate_items), user_id)
        scores = self.predict_scores(user_ids, candidate_items)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(int(candidate_items[i]), float(scores[i])) for i in top_indices]


# ==================== DeepFM ====================

class DeepFMNetwork(nn.Module):
    """
    DeepFM: A Factorization-Machine based Neural Network.
    
    Combines FM (2nd order feature interactions) with DNN.
    Reference: Guo et al. "DeepFM" (IJCAI 2017)
    """
    
    def __init__(
        self,
        num_fields: int,
        num_features: int,
        embedding_dim: int = 10,
        mlp_hidden_dims: List[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_fields = num_fields
        mlp_hidden_dims = mlp_hidden_dims or [256, 128, 64]
        
        # First-order (linear) weights
        self.linear = nn.Embedding(num_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # FM embeddings (for 2nd order interactions)
        self.fm_embedding = nn.Embedding(num_features, embedding_dim)
        
        # Deep component
        deep_input_dim = num_fields * embedding_dim
        layers = []
        prev_dim = deep_input_dim
        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.normal_(self.fm_embedding.weight, mean=0, std=0.01)
    
    def forward(self, feature_indices: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feature_indices: (batch_size, num_fields) feature indices
            feature_values: (batch_size, num_fields) feature values
        
        Returns:
            (batch_size,) prediction scores
        """
        # Linear component
        linear_out = (self.linear(feature_indices).squeeze(-1) * feature_values).sum(dim=1)
        
        # FM component (2nd order interactions)
        fm_emb = self.fm_embedding(feature_indices) * feature_values.unsqueeze(-1)
        
        # Efficient FM: (sum)^2 - sum(square) for all pairs
        sum_square = fm_emb.sum(dim=1) ** 2
        square_sum = (fm_emb ** 2).sum(dim=1)
        fm_out = 0.5 * (sum_square - square_sum).sum(dim=1)
        
        # Deep component
        deep_input = fm_emb.view(fm_emb.size(0), -1)
        deep_out = self.deep(deep_input).squeeze(-1)
        
        # Combine
        output = self.bias + linear_out + fm_out + deep_out
        
        return torch.sigmoid(output)


class DeepFM(BaseNeuralNetwork):
    """
    DeepFM model for CTR prediction and ranking.
    """
    
    MODEL_TYPE = "recommendation"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "deepfm",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_fields": 20,
            "num_features": 100000,
            "embedding_dim": 10,
            "mlp_hidden_dims": [256, 128, 64],
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 1024,
            "epochs": 10,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return DeepFMNetwork(
            num_fields=self.config["num_fields"],
            num_features=self.config["num_features"],
            embedding_dim=self.config["embedding_dim"],
            mlp_hidden_dims=self.config["mlp_hidden_dims"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        return nn.BCELoss()


# ==================== Two-Tower Model ====================

class TwoTowerNetwork(nn.Module):
    """
    Two-Tower Model for large-scale retrieval.
    
    Separately encodes users and items, enabling efficient ANN search.
    Reference: Used by YouTube, Pinterest, etc.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_user_features: int,
        num_item_features: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [256, 128]
        
        # User tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        user_layers = []
        user_input_dim = embedding_dim + num_user_features
        prev_dim = user_input_dim
        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        user_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # Item tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        item_layers = []
        item_input_dim = embedding_dim + num_item_features
        prev_dim = item_input_dim
        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        item_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def encode_user(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode users to embeddings."""
        user_emb = self.user_embedding(user_ids)
        combined = torch.cat([user_emb, user_features], dim=-1)
        output = self.user_tower(combined)
        return F.normalize(output, p=2, dim=-1)
    
    def encode_item(
        self,
        item_ids: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode items to embeddings."""
        item_emb = self.item_embedding(item_ids)
        combined = torch.cat([item_emb, item_features], dim=-1)
        output = self.item_tower(combined)
        return F.normalize(output, p=2, dim=-1)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores.
        
        Returns:
            (batch_size,) similarity scores
        """
        user_emb = self.encode_user(user_ids, user_features)
        item_emb = self.encode_item(item_ids, item_features)
        
        # Cosine similarity with temperature
        similarity = (user_emb * item_emb).sum(dim=-1) / self.temperature
        
        return similarity


class TwoTowerRecommender(BaseNeuralNetwork):
    """
    Two-Tower model for scalable retrieval.
    """
    
    MODEL_TYPE = "recommendation"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "two_tower",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_users": 100000,
            "num_items": 50000,
            "num_user_features": 50,
            "num_item_features": 100,
            "embedding_dim": 128,
            "hidden_dims": [256, 128],
            "learning_rate": 0.001,
            "batch_size": 512,
            "epochs": 20,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
        
        # Cache for item embeddings
        self._item_embeddings_cache = None
    
    def build_model(self) -> nn.Module:
        return TwoTowerNetwork(
            num_users=self.config["num_users"],
            num_items=self.config["num_items"],
            num_user_features=self.config["num_user_features"],
            num_item_features=self.config["num_item_features"],
            embedding_dim=self.config["embedding_dim"],
            hidden_dims=self.config["hidden_dims"],
        )
    
    def get_loss_function(self) -> nn.Module:
        # In-batch negatives with cross entropy
        return nn.CrossEntropyLoss()
    
    def build_item_index(
        self,
        item_ids: np.ndarray,
        item_features: np.ndarray,
        batch_size: int = 1024
    ):
        """Pre-compute item embeddings for fast retrieval."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(item_ids), batch_size):
                ids = torch.LongTensor(item_ids[i:i + batch_size]).to(self.device)
                features = torch.FloatTensor(item_features[i:i + batch_size]).to(self.device)
                emb = self.model.encode_item(ids, features)
                embeddings.append(emb.cpu().numpy())
        
        self._item_embeddings_cache = {
            "item_ids": item_ids,
            "embeddings": np.vstack(embeddings)
        }
        
        logger.info(f"Built item index with {len(item_ids)} items")
    
    def retrieve(
        self,
        user_id: int,
        user_features: np.ndarray,
        top_k: int = 100
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-K items for a user using ANN.
        
        Args:
            user_id: User ID
            user_features: User feature vector
            top_k: Number of items to retrieve
        
        Returns:
            List of (item_id, score) tuples
        """
        if self._item_embeddings_cache is None:
            raise ValueError("Item index not built. Call build_item_index first.")
        
        self.model.eval()
        
        with torch.no_grad():
            user_ids = torch.LongTensor([user_id]).to(self.device)
            features = torch.FloatTensor([user_features]).to(self.device)
            user_emb = self.model.encode_user(user_ids, features).cpu().numpy()
        
        # Compute similarities
        item_embs = self._item_embeddings_cache["embeddings"]
        similarities = np.dot(item_embs, user_emb.T).flatten()
        
        # Get top-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        item_ids = self._item_embeddings_cache["item_ids"]
        
        return [(int(item_ids[i]), float(similarities[i])) for i in top_indices]


# ==================== Sequential Recommender ====================

class SequentialRecommenderNetwork(nn.Module):
    """
    Sequential Recommender using Transformer architecture.
    
    Similar to BERT4Rec but with causal attention for autoregressive generation.
    """
    
    def __init__(
        self,
        num_items: int,
        max_seq_length: int = 50,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        
        # Item embedding (with padding index)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(embedding_dim, num_items)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _generate_causal_mask(self, seq_length: int, device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device) * float('-inf'),
            diagonal=1
        )
        return mask
    
    def forward(
        self,
        item_sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            item_sequences: (batch_size, seq_length) item indices (1-indexed, 0 = padding)
            attention_mask: (batch_size, seq_length) attention mask
        
        Returns:
            (batch_size, seq_length, num_items) logits for next item prediction
        """
        batch_size, seq_length = item_sequences.shape
        
        # Get embeddings
        item_emb = self.item_embedding(item_sequences)
        
        positions = torch.arange(seq_length, device=item_sequences.device)
        pos_emb = self.position_embedding(positions)
        
        hidden = self.dropout(self.layer_norm(item_emb + pos_emb))
        
        # Causal mask
        causal_mask = self._generate_causal_mask(seq_length, item_sequences.device)
        
        # Padding mask
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Output
        logits = self.output_layer(hidden)
        
        return logits
    
    def predict_next(self, item_sequence: torch.Tensor) -> torch.Tensor:
        """Predict next item probabilities."""
        logits = self.forward(item_sequence)
        # Get last position
        return F.softmax(logits[:, -1, :], dim=-1)


class SequenceRecommender(BaseNeuralNetwork):
    """
    Sequence-based recommender using Transformer.
    """
    
    MODEL_TYPE = "recommendation"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "sequence_recommender",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_items": 50000,
            "max_seq_length": 50,
            "embedding_dim": 128,
            "num_heads": 4,
            "num_layers": 4,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 30,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return SequentialRecommenderNetwork(
            num_items=self.config["num_items"],
            max_seq_length=self.config["max_seq_length"],
            embedding_dim=self.config["embedding_dim"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    def recommend_next(
        self,
        item_history: List[int],
        top_k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommend next items based on sequence.
        
        Args:
            item_history: List of item IDs (most recent last)
            top_k: Number of recommendations
            exclude_items: Items to exclude
        
        Returns:
            List of (item_id, probability) tuples
        """
        self.model.eval()
        max_len = self.config["max_seq_length"]
        
        # Prepare sequence (add 1 for 1-indexing)
        sequence = [i + 1 for i in item_history[-max_len:]]
        
        # Pad if needed
        if len(sequence) < max_len:
            sequence = [0] * (max_len - len(sequence)) + sequence
        
        with torch.no_grad():
            seq_tensor = torch.LongTensor([sequence]).to(self.device)
            probs = self.model.predict_next(seq_tensor).cpu().numpy()[0]
        
        # Exclude items
        if exclude_items:
            for item_id in exclude_items:
                if 0 <= item_id < len(probs):
                    probs[item_id] = 0
        
        # Get top-K
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        return [(int(idx), float(probs[idx])) for idx in top_indices]
