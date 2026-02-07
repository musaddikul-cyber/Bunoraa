"""
Embedding Models for Products and Users

Provides learned vector representations for:
- Products (using product attributes, descriptions, images)
- Users (using behavior, preferences, demographics)
- Categories
- Tags

These embeddings enable:
- Similarity search
- Recommendation systems
- Clustering
- Visualization
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

logger = logging.getLogger("bunoraa.ml.embeddings")


class ProductEmbeddingNetwork(nn.Module):
    """
    Neural network for learning product embeddings.
    
    Architecture:
    - Category embedding
    - Text encoder (for name/description)
    - Numerical features MLP
    - Fusion layer
    """
    
    def __init__(
        self,
        num_products: int,
        num_categories: int,
        num_tags: int,
        embedding_dim: int = 128,
        category_embedding_dim: int = 32,
        tag_embedding_dim: int = 16,
        num_numerical_features: int = 10,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        hidden_dims = hidden_dims or [256, 128]
        
        # Learnable embeddings
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim)
        self.tag_embedding = nn.Embedding(num_tags, tag_embedding_dim)
        
        # Numerical features MLP
        self.numerical_encoder = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
        )
        
        # Fusion network
        fusion_input_dim = embedding_dim + category_embedding_dim + tag_embedding_dim + 32
        
        layers = []
        prev_dim = fusion_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.fusion = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        product_ids: torch.Tensor,
        category_ids: torch.Tensor,
        tag_ids: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute product embeddings.
        
        Args:
            product_ids: (batch_size,) product indices
            category_ids: (batch_size,) primary category indices
            tag_ids: (batch_size, num_tags) multi-hot tag indices or aggregated tag ID
            numerical_features: (batch_size, num_numerical_features)
        
        Returns:
            (batch_size, embedding_dim) product embeddings
        """
        # Get base embeddings
        prod_emb = self.product_embedding(product_ids)
        cat_emb = self.category_embedding(category_ids)
        
        # Handle tag embeddings (mean pool if multiple)
        if tag_ids.dim() == 2:
            tag_emb = self.tag_embedding(tag_ids)
            tag_emb = tag_emb.mean(dim=1)
        else:
            tag_emb = self.tag_embedding(tag_ids)
        
        # Encode numerical features
        num_emb = self.numerical_encoder(numerical_features)
        
        # Fuse all representations
        combined = torch.cat([prod_emb, cat_emb, tag_emb, num_emb], dim=-1)
        output = self.fusion(combined)
        
        # L2 normalize for cosine similarity
        output = F.normalize(output, p=2, dim=-1)
        
        return output
    
    def get_product_embedding(self, product_id: int) -> torch.Tensor:
        """Get embedding for a single product."""
        return self.product_embedding.weight[product_id]


class ProductEmbeddingModel(BaseNeuralNetwork):
    """
    Product embedding model with training and inference capabilities.
    
    Learns embeddings via:
    - Triplet loss (anchor, positive, negative)
    - Contrastive learning
    - Multi-task learning (category prediction, etc.)
    """
    
    MODEL_TYPE = "embedding"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "product_embeddings",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_products": 10000,
            "num_categories": 100,
            "num_tags": 500,
            "embedding_dim": 128,
            "category_embedding_dim": 32,
            "tag_embedding_dim": 16,
            "num_numerical_features": 10,
            "hidden_dims": [256, 128],
            "margin": 0.5,  # Triplet loss margin
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 50,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        """Build the product embedding network."""
        return ProductEmbeddingNetwork(
            num_products=self.config["num_products"],
            num_categories=self.config["num_categories"],
            num_tags=self.config["num_tags"],
            embedding_dim=self.config["embedding_dim"],
            category_embedding_dim=self.config["category_embedding_dim"],
            tag_embedding_dim=self.config["tag_embedding_dim"],
            num_numerical_features=self.config["num_numerical_features"],
            hidden_dims=self.config["hidden_dims"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Triplet margin loss for embedding learning."""
        return nn.TripletMarginLoss(margin=self.config["margin"])
    
    def fit_triplets(
        self,
        triplets: List[Tuple[Dict, Dict, Dict]],
        validation_triplets: Optional[List[Tuple[Dict, Dict, Dict]]] = None,
        **kwargs
    ):
        """
        Train on triplet data (anchor, positive, negative).
        
        Each triplet contains dicts with keys:
        - product_id, category_id, tag_ids, numerical_features
        """
        if self.model is None:
            self.model = self.build_model()
        
        self.model = self.model.to(self.device)
        self.model.train()
        
        epochs = kwargs.get("epochs", self.config["epochs"])
        batch_size = kwargs.get("batch_size", self.config["batch_size"])
        
        loss_fn = self.get_loss_function()
        self.optimizer = self.get_optimizer(self.model)
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                
                anchors, positives, negatives = zip(*batch)
                
                # Process each component
                anchor_emb = self._process_batch(anchors)
                positive_emb = self._process_batch(positives)
                negative_emb = self._process_batch(negatives)
                
                self.optimizer.zero_grad()
                loss = loss_fn(anchor_emb, positive_emb, negative_emb)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def _process_batch(self, items: List[Dict]) -> torch.Tensor:
        """Process a batch of items and return embeddings."""
        product_ids = torch.LongTensor([x["product_id"] for x in items]).to(self.device)
        category_ids = torch.LongTensor([x["category_id"] for x in items]).to(self.device)
        tag_ids = torch.LongTensor([x.get("tag_ids", [0]) for x in items]).to(self.device)
        numerical = torch.FloatTensor([x["numerical_features"] for x in items]).to(self.device)
        
        return self.model(product_ids, category_ids, tag_ids, numerical)
    
    def encode_products(
        self,
        product_data: List[Dict],
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Encode products to embeddings.
        
        Args:
            product_data: List of dicts with product features
            batch_size: Batch size for inference
        
        Returns:
            (num_products, embedding_dim) array
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(product_data), batch_size):
                batch = product_data[i:i + batch_size]
                emb = self._process_batch(batch)
                embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        all_embeddings: np.ndarray,
        top_k: int = 10,
        exclude_indices: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Find most similar products by cosine similarity.
        
        Args:
            query_embedding: (embedding_dim,) query vector
            all_embeddings: (num_products, embedding_dim) all product embeddings
            top_k: Number of results
            exclude_indices: Indices to exclude from results
        
        Returns:
            List of (index, similarity_score) tuples
        """
        # Compute cosine similarities
        query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8
        normalized = all_embeddings / norms
        
        similarities = np.dot(normalized, query)
        
        # Exclude specified indices
        if exclude_indices:
            for idx in exclude_indices:
                similarities[idx] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class UserEmbeddingNetwork(nn.Module):
    """
    Neural network for learning user embeddings.
    
    Captures user behavior and preferences through:
    - Interaction history encoding (products viewed/purchased)
    - Temporal patterns
    - User attributes
    """
    
    def __init__(
        self,
        num_users: int,
        product_embedding_dim: int = 128,
        user_embedding_dim: int = 128,
        num_user_features: int = 20,
        max_history_length: int = 50,
        num_attention_heads: int = 4,
        num_transformer_layers: int = 2
    ):
        super().__init__()
        
        self.user_embedding_dim = user_embedding_dim
        self.max_history_length = max_history_length
        
        # Direct user embedding
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        
        # User features encoder
        self.user_features_encoder = nn.Sequential(
            nn.Linear(num_user_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, user_embedding_dim // 2),
        )
        
        # Position encoding for history sequence
        self.position_embedding = nn.Embedding(max_history_length, product_embedding_dim)
        
        # Transformer for encoding interaction history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=product_embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=product_embedding_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Attention pooling for history
        self.attention_pool = nn.Sequential(
            nn.Linear(product_embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Fusion layer
        fusion_dim = user_embedding_dim + user_embedding_dim // 2 + product_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, user_embedding_dim * 2),
            nn.LayerNorm(user_embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(user_embedding_dim * 2, user_embedding_dim),
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
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute user embeddings.
        
        Args:
            user_ids: (batch_size,) user indices
            user_features: (batch_size, num_user_features)
            history_embeddings: (batch_size, seq_len, product_embedding_dim)
            history_mask: (batch_size, seq_len) mask for padding
        
        Returns:
            (batch_size, user_embedding_dim) user embeddings
        """
        batch_size, seq_len, _ = history_embeddings.shape
        
        # Direct user embedding
        user_emb = self.user_embedding(user_ids)
        
        # User features embedding
        user_feat_emb = self.user_features_encoder(user_features)
        
        # Add position embeddings to history
        positions = torch.arange(seq_len, device=history_embeddings.device)
        pos_emb = self.position_embedding(positions)
        history_with_pos = history_embeddings + pos_emb.unsqueeze(0)
        
        # Encode history with transformer
        if history_mask is not None:
            # Convert to attention mask format
            src_key_padding_mask = ~history_mask.bool()
        else:
            src_key_padding_mask = None
        
        encoded_history = self.history_encoder(
            history_with_pos,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Attention pooling
        attention_weights = self.attention_pool(encoded_history)
        if history_mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~history_mask.unsqueeze(-1).bool(),
                float('-inf')
            )
        attention_weights = F.softmax(attention_weights, dim=1)
        history_emb = (encoded_history * attention_weights).sum(dim=1)
        
        # Fuse all representations
        combined = torch.cat([user_emb, user_feat_emb, history_emb], dim=-1)
        output = self.fusion(combined)
        
        # L2 normalize
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class UserEmbeddingModel(BaseNeuralNetwork):
    """
    User embedding model with training and inference capabilities.
    """
    
    MODEL_TYPE = "embedding"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "user_embeddings",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_users": 100000,
            "product_embedding_dim": 128,
            "user_embedding_dim": 128,
            "num_user_features": 20,
            "max_history_length": 50,
            "num_attention_heads": 4,
            "num_transformer_layers": 2,
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 30,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
        
        # Reference to product embedding model
        self._product_embedder = None
    
    def set_product_embedder(self, embedder: ProductEmbeddingModel):
        """Set the product embedding model for encoding history."""
        self._product_embedder = embedder
    
    def build_model(self) -> nn.Module:
        return UserEmbeddingNetwork(
            num_users=self.config["num_users"],
            product_embedding_dim=self.config["product_embedding_dim"],
            user_embedding_dim=self.config["user_embedding_dim"],
            num_user_features=self.config["num_user_features"],
            max_history_length=self.config["max_history_length"],
            num_attention_heads=self.config["num_attention_heads"],
            num_transformer_layers=self.config["num_transformer_layers"],
        )
    
    def get_loss_function(self) -> nn.Module:
        return nn.TripletMarginLoss(margin=0.5)
    
    def encode_users(
        self,
        user_data: List[Dict],
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Encode users to embeddings.
        
        Args:
            user_data: List of dicts with:
                - user_id: int
                - user_features: List[float]
                - history_embeddings: (seq_len, embedding_dim) or List of product embeddings
                - history_mask: Optional mask
        
        Returns:
            (num_users, user_embedding_dim) array
        """
        self.model.eval()
        embeddings = []
        
        max_len = self.config["max_history_length"]
        emb_dim = self.config["product_embedding_dim"]
        
        with torch.no_grad():
            for i in range(0, len(user_data), batch_size):
                batch = user_data[i:i + batch_size]
                
                user_ids = torch.LongTensor([x["user_id"] for x in batch]).to(self.device)
                user_features = torch.FloatTensor([x["user_features"] for x in batch]).to(self.device)
                
                # Pad history to max length
                history_list = []
                mask_list = []
                for x in batch:
                    hist = np.array(x["history_embeddings"])
                    if len(hist) > max_len:
                        hist = hist[-max_len:]  # Keep most recent
                    elif len(hist) < max_len:
                        padding = np.zeros((max_len - len(hist), emb_dim))
                        mask = [1] * len(hist) + [0] * (max_len - len(hist))
                        hist = np.vstack([hist, padding]) if len(hist) > 0 else padding
                    else:
                        mask = [1] * max_len
                    
                    history_list.append(hist)
                    mask_list.append(mask if len(mask) == max_len else [1] * max_len)
                
                history = torch.FloatTensor(np.array(history_list)).to(self.device)
                mask = torch.FloatTensor(np.array(mask_list)).to(self.device)
                
                emb = self.model(user_ids, user_features, history, mask)
                embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
