"""
Semantic Search and NLP Models

Models for:
- Semantic search
- Query understanding
- Text similarity
- Search ranking
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

logger = logging.getLogger("bunoraa.ml.search")


class QueryEncoderNetwork(nn.Module):
    """
    Neural encoder for search queries.
    
    Converts text queries to dense vectors for semantic search.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_query_length: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_query_length = max_query_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_query_length, embedding_dim)
        
        # Query encoder (bidirectional LSTM)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode query to vector.
        
        Args:
            token_ids: (batch_size, seq_len) token indices
            attention_mask: (batch_size, seq_len) mask
        
        Returns:
            (batch_size, embedding_dim) query embeddings
        """
        batch_size, seq_len = token_ids.shape
        
        # Get embeddings
        token_emb = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.position_embedding(positions)
        
        hidden = token_emb + pos_emb
        
        # Encode with LSTM
        lstm_output, _ = self.encoder(hidden)
        
        # Self-attention
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        attn_output, _ = self.attention(
            lstm_output, lstm_output, lstm_output,
            key_padding_mask=key_padding_mask
        )
        
        # Pool (mean of non-padded tokens)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (attn_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = attn_output.mean(dim=1)
        
        # Project to output dimension
        output = self.output_proj(pooled)
        
        # L2 normalize
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class QueryEncoder(BaseNeuralNetwork):
    """
    Query encoder for semantic search.
    """
    
    MODEL_TYPE = "search"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "query_encoder",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "vocab_size": 50000,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "max_query_length": 50,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 20,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
        
        # Simple tokenizer (can be replaced with better tokenizer)
        self._vocab = {}
        self._reverse_vocab = {}
    
    def build_model(self) -> nn.Module:
        return QueryEncoderNetwork(
            vocab_size=self.config["vocab_size"],
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            max_query_length=self.config["max_query_length"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Contrastive loss for query-product matching."""
        class ContrastiveLoss(nn.Module):
            def __init__(self, temperature=0.07):
                super().__init__()
                self.temperature = temperature
            
            def forward(self, query_emb, pos_emb, neg_emb=None):
                # Positive similarity
                pos_sim = (query_emb * pos_emb).sum(dim=-1) / self.temperature
                
                if neg_emb is not None:
                    # In-batch negatives + explicit negatives
                    neg_sim = torch.matmul(query_emb, neg_emb.T) / self.temperature
                    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
                    labels = torch.zeros(len(query_emb), dtype=torch.long, device=query_emb.device)
                    return F.cross_entropy(logits, labels)
                else:
                    # Simple contrastive with in-batch negatives
                    logits = torch.matmul(query_emb, pos_emb.T) / self.temperature
                    labels = torch.arange(len(query_emb), device=query_emb.device)
                    return F.cross_entropy(logits, labels)
        
        return ContrastiveLoss()
    
    def tokenize(self, query: str) -> List[int]:
        """Simple tokenization (word-level)."""
        tokens = query.lower().split()[:self.config["max_query_length"]]
        
        ids = []
        for token in tokens:
            if token not in self._vocab:
                if len(self._vocab) < self.config["vocab_size"] - 2:
                    token_id = len(self._vocab) + 2  # Reserve 0 for padding, 1 for UNK
                    self._vocab[token] = token_id
                    self._reverse_vocab[token_id] = token
                else:
                    token_id = 1  # UNK
            else:
                token_id = self._vocab[token]
            ids.append(token_id)
        
        return ids
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to vector."""
        self.model.eval()
        
        token_ids = self.tokenize(query)
        max_len = self.config["max_query_length"]
        
        # Pad
        if len(token_ids) < max_len:
            mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        else:
            mask = [1] * max_len
        
        with torch.no_grad():
            ids = torch.LongTensor([token_ids]).to(self.device)
            mask_tensor = torch.FloatTensor([mask]).to(self.device)
            embedding = self.model(ids, mask_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def encode_queries(self, queries: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode multiple queries."""
        self.model.eval()
        embeddings = []
        
        max_len = self.config["max_query_length"]
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            all_ids = []
            all_masks = []
            
            for query in batch_queries:
                token_ids = self.tokenize(query)
                
                if len(token_ids) < max_len:
                    mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
                    token_ids = token_ids + [0] * (max_len - len(token_ids))
                else:
                    mask = [1] * max_len
                    token_ids = token_ids[:max_len]
                
                all_ids.append(token_ids)
                all_masks.append(mask)
            
            with torch.no_grad():
                ids = torch.LongTensor(all_ids).to(self.device)
                masks = torch.FloatTensor(all_masks).to(self.device)
                batch_emb = self.model(ids, masks)
                embeddings.append(batch_emb.cpu().numpy())
        
        return np.vstack(embeddings)


class SemanticSearchNetwork(nn.Module):
    """
    Full semantic search model with query and product encoders.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        num_products: int = 100000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_query_length: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Query encoder
        self.query_encoder = QueryEncoderNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_query_length=max_query_length,
            dropout=dropout
        )
        
        # Product encoder (simpler - uses pre-computed features)
        self.product_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # Score calibration
        self.score_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64),  # query, product, query*product
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # Temperature for similarity
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
    
    def encode_query(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode query."""
        return self.query_encoder(token_ids, attention_mask)
    
    def encode_product(self, product_features: torch.Tensor) -> torch.Tensor:
        """Encode product from features."""
        output = self.product_proj(product_features)
        return F.normalize(output, p=2, dim=-1)
    
    def forward(
        self,
        query_token_ids: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        product_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance scores.
        
        Args:
            query_token_ids: (batch_size, seq_len)
            query_mask: (batch_size, seq_len)
            product_features: (batch_size, embedding_dim)
        
        Returns:
            (batch_size,) relevance scores
        """
        query_emb = self.encode_query(query_token_ids, query_mask)
        product_emb = self.encode_product(product_features)
        
        # Cosine similarity
        sim = (query_emb * product_emb).sum(dim=-1) / self.temperature
        
        # Additional scoring with interaction features
        interaction = query_emb * product_emb
        combined = torch.cat([query_emb, product_emb, interaction], dim=-1)
        calibrated = self.score_head(combined).squeeze(-1)
        
        # Combine
        score = sim + 0.1 * calibrated
        
        return score


class SemanticSearchModel(BaseNeuralNetwork):
    """
    Semantic search model for product retrieval.
    """
    
    MODEL_TYPE = "search"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "semantic_search",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "vocab_size": 50000,
            "num_products": 100000,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "max_query_length": 50,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 30,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
        
        # Product index
        self._product_index = None
        self._product_ids = None
        
        # Vocab
        self._vocab = {}
    
    def build_model(self) -> nn.Module:
        return SemanticSearchNetwork(
            vocab_size=self.config["vocab_size"],
            num_products=self.config["num_products"],
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            max_query_length=self.config["max_query_length"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Ranking loss."""
        class RankingLoss(nn.Module):
            def __init__(self, margin=0.2):
                super().__init__()
                self.margin = margin
            
            def forward(self, pos_scores, neg_scores):
                # Margin ranking loss
                loss = F.relu(self.margin - pos_scores + neg_scores)
                return loss.mean()
        
        return RankingLoss()
    
    def tokenize(self, query: str) -> List[int]:
        """Tokenize query."""
        tokens = query.lower().split()[:self.config["max_query_length"]]
        
        ids = []
        for token in tokens:
            token_id = self._vocab.get(token, 1)  # 1 = UNK
            ids.append(token_id)
        
        return ids
    
    def build_product_index(
        self,
        product_ids: List[str],
        product_features: np.ndarray
    ):
        """Build product index for fast retrieval."""
        self.model.eval()
        
        embeddings = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(product_features), batch_size):
                batch = torch.FloatTensor(product_features[i:i + batch_size]).to(self.device)
                emb = self.model.encode_product(batch)
                embeddings.append(emb.cpu().numpy())
        
        self._product_index = np.vstack(embeddings)
        self._product_ids = product_ids
        
        logger.info(f"Built product index with {len(product_ids)} products")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for products.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters (not applied in embedding search)
        
        Returns:
            List of search results
        """
        if self._product_index is None:
            raise ValueError("Product index not built. Call build_product_index first.")
        
        self.model.eval()
        
        # Encode query
        token_ids = self.tokenize(query)
        max_len = self.config["max_query_length"]
        
        if len(token_ids) < max_len:
            mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        else:
            mask = [1] * max_len
            token_ids = token_ids[:max_len]
        
        with torch.no_grad():
            ids = torch.LongTensor([token_ids]).to(self.device)
            mask_tensor = torch.FloatTensor([mask]).to(self.device)
            query_emb = self.model.encode_query(ids, mask_tensor).cpu().numpy()[0]
        
        # Compute similarities
        similarities = np.dot(self._product_index, query_emb)
        
        # Get top-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "product_id": self._product_ids[idx],
                "score": float(similarities[idx]),
                "rank": len(results) + 1
            })
        
        return results
    
    def rerank(
        self,
        query: str,
        product_features: List[np.ndarray],
        product_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rerank products for a query.
        
        Args:
            query: Search query
            product_features: List of product feature vectors
            product_ids: List of product IDs
        
        Returns:
            Reranked list of products
        """
        self.model.eval()
        
        # Encode query
        token_ids = self.tokenize(query)
        max_len = self.config["max_query_length"]
        
        if len(token_ids) < max_len:
            mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        else:
            mask = [1] * max_len
            token_ids = token_ids[:max_len]
        
        # Compute scores for each product
        scores = []
        
        with torch.no_grad():
            ids = torch.LongTensor([token_ids]).to(self.device)
            mask_tensor = torch.FloatTensor([mask]).to(self.device)
            
            for features in product_features:
                prod = torch.FloatTensor([features]).to(self.device)
                score = self.model(ids, mask_tensor, prod).item()
                scores.append(score)
        
        # Sort by score
        ranked_indices = np.argsort(scores)[::-1]
        
        results = []
        for rank, idx in enumerate(ranked_indices):
            results.append({
                "product_id": product_ids[idx],
                "score": scores[idx],
                "rank": rank + 1
            })
        
        return results
