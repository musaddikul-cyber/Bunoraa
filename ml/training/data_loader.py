"""
Data Loading Utilities

Factory classes and utilities for creating data loaders.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import random

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

logger = logging.getLogger("bunoraa.ml.training")


@dataclass
class DataConfig:
    """Configuration for data loading."""
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False
    
    # Sampling
    negative_samples: int = 5
    hard_negative_ratio: float = 0.3
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Temporal
    time_window_days: int = 90
    sequence_length: int = 50


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        transform: Optional[Any] = None
    ):
        self.data = data
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        if self.transform:
            item = self.transform(item)
        
        return item


class InteractionDataset(Dataset):
    """Dataset for user-product interactions."""
    
    def __init__(
        self,
        user_ids: np.ndarray,
        product_ids: np.ndarray,
        labels: np.ndarray,
        user_features: Optional[np.ndarray] = None,
        product_features: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        negative_sampler: Optional[Any] = None
    ):
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.labels = labels
        self.user_features = user_features
        self.product_features = product_features
        self.timestamps = timestamps
        self.negative_sampler = negative_sampler
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "user_id": self.user_ids[idx],
            "product_id": self.product_ids[idx],
            "label": self.labels[idx],
        }
        
        if self.user_features is not None:
            item["user_features"] = self.user_features[idx]
        
        if self.product_features is not None:
            item["product_features"] = self.product_features[idx]
        
        if self.timestamps is not None:
            item["timestamp"] = self.timestamps[idx]
        
        # Sample negatives if available
        if self.negative_sampler is not None:
            neg_products = self.negative_sampler.sample(
                self.user_ids[idx],
                exclude=[self.product_ids[idx]]
            )
            item["negative_products"] = neg_products
        
        return item


class SequenceDataset(Dataset):
    """Dataset for sequential recommendations."""
    
    def __init__(
        self,
        sequences: List[List[int]],
        max_length: int = 50,
        mask_prob: float = 0.15
    ):
        self.sequences = sequences
        self.max_length = max_length
        self.mask_prob = mask_prob
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sequence = self.sequences[idx]
        
        # Truncate or pad
        if len(sequence) > self.max_length:
            sequence = sequence[-self.max_length:]
        
        seq_len = len(sequence)
        
        # Pad
        padding_length = self.max_length - seq_len
        padded_seq = [0] * padding_length + sequence
        attention_mask = [0] * padding_length + [1] * seq_len
        
        # Target is next item prediction
        if seq_len > 1:
            input_seq = padded_seq[:-1] + [0]  # Shift right, last position is masked
            target = padded_seq[-1]
        else:
            input_seq = padded_seq
            target = 0
        
        return {
            "input_ids": np.array(input_seq, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.float32),
            "target": target,
        }


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        features: Optional[np.ndarray] = None,
        lookback: int = 30,
        horizon: int = 7
    ):
        self.values = values
        self.timestamps = timestamps
        self.features = features
        self.lookback = lookback
        self.horizon = horizon
        
        # Create valid indices
        self.valid_indices = list(range(lookback, len(values) - horizon + 1))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_idx = self.valid_indices[idx]
        
        # Input sequence
        input_values = self.values[start_idx - self.lookback:start_idx]
        
        # Target sequence
        target_values = self.values[start_idx:start_idx + self.horizon]
        
        item = {
            "input_values": input_values.astype(np.float32),
            "target_values": target_values.astype(np.float32),
        }
        
        if self.features is not None:
            input_features = self.features[start_idx - self.lookback:start_idx]
            item["input_features"] = input_features.astype(np.float32)
        
        return item


class NegativeSampler:
    """Efficient negative sampling for contrastive learning."""
    
    def __init__(
        self,
        num_items: int,
        user_interactions: Dict[int, set],
        popularity: Optional[np.ndarray] = None,
        num_negatives: int = 5
    ):
        self.num_items = num_items
        self.user_interactions = user_interactions
        self.num_negatives = num_negatives
        
        # Popularity-based sampling distribution
        if popularity is not None:
            # Smooth popularity with power of 0.75 (word2vec style)
            self.sample_probs = popularity ** 0.75
            self.sample_probs /= self.sample_probs.sum()
        else:
            self.sample_probs = None
    
    def sample(
        self,
        user_id: int,
        exclude: Optional[List[int]] = None,
        num_samples: Optional[int] = None
    ) -> List[int]:
        """Sample negative items for a user."""
        num_samples = num_samples or self.num_negatives
        
        # Items to exclude
        exclude_set = self.user_interactions.get(user_id, set())
        if exclude:
            exclude_set = exclude_set.union(set(exclude))
        
        negatives = []
        max_attempts = num_samples * 10
        attempts = 0
        
        while len(negatives) < num_samples and attempts < max_attempts:
            if self.sample_probs is not None:
                candidate = np.random.choice(self.num_items, p=self.sample_probs)
            else:
                candidate = random.randint(0, self.num_items - 1)
            
            if candidate not in exclude_set:
                negatives.append(candidate)
            
            attempts += 1
        
        # Fill with random if needed
        while len(negatives) < num_samples:
            negatives.append(random.randint(0, self.num_items - 1))
        
        return negatives
    
    def sample_batch(
        self,
        user_ids: List[int],
        exclude: Optional[List[List[int]]] = None
    ) -> np.ndarray:
        """Sample negatives for a batch of users."""
        batch_negatives = []
        
        for i, user_id in enumerate(user_ids):
            user_exclude = exclude[i] if exclude else None
            negatives = self.sample(user_id, user_exclude)
            batch_negatives.append(negatives)
        
        return np.array(batch_negatives)


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_interaction_loader(
        interactions: List[Dict[str, Any]],
        config: DataConfig,
        user_features: Optional[Dict[int, np.ndarray]] = None,
        product_features: Optional[Dict[int, np.ndarray]] = None,
        mode: str = "train"
    ) -> DataLoader:
        """Create data loader for interaction data."""
        # Extract arrays
        user_ids = np.array([i["user_id"] for i in interactions])
        product_ids = np.array([i["product_id"] for i in interactions])
        labels = np.array([i.get("label", 1) for i in interactions])
        
        # Optional arrays
        user_feat_arr = None
        product_feat_arr = None
        timestamps = None
        
        if user_features:
            user_feat_arr = np.array([
                user_features.get(uid, np.zeros(128))
                for uid in user_ids
            ])
        
        if product_features:
            product_feat_arr = np.array([
                product_features.get(pid, np.zeros(128))
                for pid in product_ids
            ])
        
        if "timestamp" in interactions[0]:
            timestamps = np.array([i["timestamp"] for i in interactions])
        
        # Create negative sampler for training
        negative_sampler = None
        if mode == "train":
            # Build user interaction history
            user_interactions = {}
            for uid, pid in zip(user_ids, product_ids):
                if uid not in user_interactions:
                    user_interactions[uid] = set()
                user_interactions[uid].add(pid)
            
            # Product popularity
            unique, counts = np.unique(product_ids, return_counts=True)
            popularity = np.zeros(max(product_ids) + 1)
            popularity[unique] = counts
            
            negative_sampler = NegativeSampler(
                num_items=max(product_ids) + 1,
                user_interactions=user_interactions,
                popularity=popularity,
                num_negatives=config.negative_samples
            )
        
        dataset = InteractionDataset(
            user_ids=user_ids,
            product_ids=product_ids,
            labels=labels,
            user_features=user_feat_arr,
            product_features=product_feat_arr,
            timestamps=timestamps,
            negative_sampler=negative_sampler
        )
        
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle and mode == "train",
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last and mode == "train",
        )
    
    @staticmethod
    def create_sequence_loader(
        sequences: List[List[int]],
        config: DataConfig,
        mode: str = "train"
    ) -> DataLoader:
        """Create data loader for sequence data."""
        dataset = SequenceDataset(
            sequences=sequences,
            max_length=config.sequence_length,
        )
        
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle and mode == "train",
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    
    @staticmethod
    def create_timeseries_loader(
        values: np.ndarray,
        timestamps: np.ndarray,
        config: DataConfig,
        features: Optional[np.ndarray] = None,
        lookback: int = 30,
        horizon: int = 7,
        mode: str = "train"
    ) -> DataLoader:
        """Create data loader for time series data."""
        dataset = TimeSeriesDataset(
            values=values,
            timestamps=timestamps,
            features=features,
            lookback=lookback,
            horizon=horizon,
        )
        
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle and mode == "train",
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )


class DatasetBuilder:
    """Builder for creating datasets from raw data."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def build_recommendation_dataset(
        self,
        interactions: List[Dict[str, Any]],
        split: bool = True
    ) -> Union[Tuple[DataLoader, DataLoader, DataLoader], DataLoader]:
        """Build dataset for recommendation training."""
        # Sort by timestamp if available
        if interactions and "timestamp" in interactions[0]:
            interactions = sorted(interactions, key=lambda x: x["timestamp"])
        
        if split:
            n = len(interactions)
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
            
            train_data = interactions[:train_end]
            val_data = interactions[train_end:val_end]
            test_data = interactions[val_end:]
            
            train_loader = DataLoaderFactory.create_interaction_loader(
                train_data, self.config, mode="train"
            )
            val_loader = DataLoaderFactory.create_interaction_loader(
                val_data, self.config, mode="val"
            )
            test_loader = DataLoaderFactory.create_interaction_loader(
                test_data, self.config, mode="test"
            )
            
            return train_loader, val_loader, test_loader
        else:
            return DataLoaderFactory.create_interaction_loader(
                interactions, self.config, mode="train"
            )
    
    def build_sequence_dataset(
        self,
        user_sequences: Dict[int, List[int]],
        split: bool = True
    ) -> Union[Tuple[DataLoader, DataLoader, DataLoader], DataLoader]:
        """Build dataset for sequential recommendations."""
        sequences = list(user_sequences.values())
        
        # Filter short sequences
        sequences = [s for s in sequences if len(s) >= 3]
        
        if split:
            random.shuffle(sequences)
            n = len(sequences)
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
            
            train_loader = DataLoaderFactory.create_sequence_loader(
                sequences[:train_end], self.config, mode="train"
            )
            val_loader = DataLoaderFactory.create_sequence_loader(
                sequences[train_end:val_end], self.config, mode="val"
            )
            test_loader = DataLoaderFactory.create_sequence_loader(
                sequences[val_end:], self.config, mode="test"
            )
            
            return train_loader, val_loader, test_loader
        else:
            return DataLoaderFactory.create_sequence_loader(
                sequences, self.config, mode="train"
            )
    
    def build_forecasting_dataset(
        self,
        time_series: Dict[str, np.ndarray],
        lookback: int = 30,
        horizon: int = 7,
        split: bool = True
    ) -> Union[Tuple[DataLoader, DataLoader, DataLoader], DataLoader]:
        """Build dataset for demand forecasting."""
        values = time_series["values"]
        timestamps = time_series["timestamps"]
        features = time_series.get("features")
        
        if split:
            n = len(values)
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
            
            train_loader = DataLoaderFactory.create_timeseries_loader(
                values[:train_end],
                timestamps[:train_end],
                self.config,
                features[:train_end] if features is not None else None,
                lookback,
                horizon,
                mode="train"
            )
            val_loader = DataLoaderFactory.create_timeseries_loader(
                values[train_end:val_end],
                timestamps[train_end:val_end],
                self.config,
                features[train_end:val_end] if features is not None else None,
                lookback,
                horizon,
                mode="val"
            )
            test_loader = DataLoaderFactory.create_timeseries_loader(
                values[val_end:],
                timestamps[val_end:],
                self.config,
                features[val_end:] if features is not None else None,
                lookback,
                horizon,
                mode="test"
            )
            
            return train_loader, val_loader, test_loader
        else:
            return DataLoaderFactory.create_timeseries_loader(
                values, timestamps, self.config, features, lookback, horizon
            )
