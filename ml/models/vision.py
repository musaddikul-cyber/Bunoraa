"""
Computer Vision Models for E-commerce

Models for:
- Product image classification
- Visual similarity search
- Image feature extraction
- Quality assessment
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

from ..core.base import BaseNeuralNetwork

logger = logging.getLogger("bunoraa.ml.vision")


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.activation(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ProductImageNetwork(nn.Module):
    """
    CNN for product image classification and feature extraction.
    
    Custom architecture optimized for e-commerce product images.
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        num_attributes: int = 50,
        embedding_dim: int = 256,
        base_channels: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Stem
        self.stem = nn.Sequential(
            ConvBlock(3, base_channels, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels, dropout),
            ResidualBlock(base_channels, dropout),
            SEBlock(base_channels),
        )
        
        # Downsample and stage 2
        self.down1 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels * 2, dropout),
            ResidualBlock(base_channels * 2, dropout),
            SEBlock(base_channels * 2),
        )
        
        # Downsample and stage 3
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels * 4, dropout),
            ResidualBlock(base_channels * 4, dropout),
            ResidualBlock(base_channels * 4, dropout),
            SEBlock(base_channels * 4),
        )
        
        # Downsample and stage 4
        self.down3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.stage4 = nn.Sequential(
            ResidualBlock(base_channels * 8, dropout),
            ResidualBlock(base_channels * 8, dropout),
            ResidualBlock(base_channels * 8, dropout),
            SEBlock(base_channels * 8),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(base_channels * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )
        
        # Attribute prediction head (multi-label)
        self.attribute_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_attributes),
        )
        
        # Quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Quality score 0-1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        features = self.feature_proj(x)
        
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch_size, 3, H, W) images
            return_features: Whether to return feature vectors
        
        Returns:
            Dictionary with predictions
        """
        features = self.extract_features(x)
        
        # Predictions
        class_logits = self.classifier(features)
        attribute_logits = self.attribute_head(features)
        quality_score = self.quality_head(features)
        
        output = {
            "class_logits": class_logits,
            "attribute_logits": attribute_logits,
            "quality_score": quality_score,
        }
        
        if return_features:
            output["features"] = F.normalize(features, p=2, dim=-1)
        
        return output


class ProductImageClassifier(BaseNeuralNetwork):
    """
    Product image classifier with multi-task learning.
    """
    
    MODEL_TYPE = "vision"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "product_classifier",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "num_classes": 100,
            "num_attributes": 50,
            "embedding_dim": 256,
            "base_channels": 64,
            "dropout": 0.2,
            "input_size": 224,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
        
        # Class names
        self._class_names = []
        self._attribute_names = []
        
        # Feature index for similarity search
        self._feature_index = None
        self._indexed_ids = None
    
    def build_model(self) -> nn.Module:
        return ProductImageNetwork(
            num_classes=self.config["num_classes"],
            num_attributes=self.config["num_attributes"],
            embedding_dim=self.config["embedding_dim"],
            base_channels=self.config["base_channels"],
            dropout=self.config["dropout"],
        )
    
    def get_loss_function(self) -> nn.Module:
        """Multi-task loss."""
        class MultiTaskLoss(nn.Module):
            def __init__(self, class_weight=1.0, attr_weight=0.5, quality_weight=0.2):
                super().__init__()
                self.class_weight = class_weight
                self.attr_weight = attr_weight
                self.quality_weight = quality_weight
                self.ce_loss = nn.CrossEntropyLoss()
                self.bce_loss = nn.BCEWithLogitsLoss()
                self.mse_loss = nn.MSELoss()
            
            def forward(self, outputs, targets):
                # Classification loss
                class_loss = self.ce_loss(
                    outputs["class_logits"],
                    targets["class_label"]
                )
                
                # Attribute loss (multi-label)
                attr_loss = self.bce_loss(
                    outputs["attribute_logits"],
                    targets["attributes"].float()
                )
                
                # Quality loss
                if "quality" in targets:
                    quality_loss = self.mse_loss(
                        outputs["quality_score"].squeeze(),
                        targets["quality"].float()
                    )
                else:
                    quality_loss = 0.0
                
                total_loss = (
                    self.class_weight * class_loss +
                    self.attr_weight * attr_loss +
                    self.quality_weight * quality_loss
                )
                
                return total_loss
        
        return MultiTaskLoss()
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: (H, W, 3) numpy array, RGB, 0-255
        
        Returns:
            (1, 3, input_size, input_size) tensor
        """
        from PIL import Image
        import torchvision.transforms as T
        
        input_size = self.config["input_size"]
        
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        tensor = transform(image).unsqueeze(0)
        
        return tensor
    
    def classify(
        self,
        image: Union[np.ndarray, torch.Tensor],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Classify a product image.
        
        Args:
            image: Image to classify
            top_k: Number of top predictions
        
        Returns:
            Classification results
        """
        self.model.eval()
        
        if isinstance(image, np.ndarray):
            image = self.preprocess_image(image)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image, return_features=True)
        
        # Get top-k classes
        probs = F.softmax(outputs["class_logits"], dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, k=top_k)
        
        predictions = []
        for i in range(top_k):
            idx = top_indices[i].item()
            class_name = self._class_names[idx] if idx < len(self._class_names) else f"class_{idx}"
            predictions.append({
                "class_id": idx,
                "class_name": class_name,
                "probability": top_probs[i].item()
            })
        
        # Get predicted attributes
        attr_probs = torch.sigmoid(outputs["attribute_logits"])[0]
        predicted_attrs = []
        for i, prob in enumerate(attr_probs):
            if prob > 0.5:
                attr_name = self._attribute_names[i] if i < len(self._attribute_names) else f"attr_{i}"
                predicted_attrs.append({
                    "attribute_id": i,
                    "attribute_name": attr_name,
                    "probability": prob.item()
                })
        
        return {
            "top_predictions": predictions,
            "attributes": predicted_attrs,
            "quality_score": outputs["quality_score"].item(),
            "features": outputs["features"].cpu().numpy()[0]
        }
    
    def extract_features(
        self,
        images: Union[List[np.ndarray], torch.Tensor],
        batch_size: int = 32
    ) -> np.ndarray:
        """Extract feature vectors for images."""
        self.model.eval()
        
        all_features = []
        
        if isinstance(images, list):
            # Process in batches
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_tensors = torch.cat([self.preprocess_image(img) for img in batch_images])
                batch_tensors = batch_tensors.to(self.device)
                
                with torch.no_grad():
                    features = self.model.extract_features(batch_tensors)
                    features = F.normalize(features, p=2, dim=-1)
                    all_features.append(features.cpu().numpy())
        else:
            images = images.to(self.device)
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                with torch.no_grad():
                    features = self.model.extract_features(batch)
                    features = F.normalize(features, p=2, dim=-1)
                    all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def build_similarity_index(
        self,
        product_ids: List[str],
        images: List[np.ndarray]
    ):
        """Build index for visual similarity search."""
        logger.info(f"Building similarity index for {len(images)} images...")
        
        features = self.extract_features(images)
        
        self._feature_index = features
        self._indexed_ids = product_ids
        
        logger.info("Similarity index built successfully")

    def save(self, path: Union[str, Path]) -> Path:
        """Save model and similarity index."""
        path = super().save(path)
        
        if self._feature_index is not None and self._indexed_ids is not None:
            index_path = path / "similarity_index.npz"
            np.savez_compressed(
                index_path,
                feature_index=self._feature_index,
                indexed_ids=self._indexed_ids
            )
            logger.info(f"Similarity index saved to {index_path}")
            
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProductImageClassifier":
        """Load model and similarity index."""
        instance = super().load(path)
        
        index_path = Path(path) / "similarity_index.npz"
        if index_path.exists():
            data = np.load(index_path, allow_pickle=True)
            instance._feature_index = data["feature_index"]
            instance._indexed_ids = data["indexed_ids"]
            logger.info(f"Similarity index loaded from {index_path}")
        
        return instance
    
    def find_similar(
        self,
        image: np.ndarray,
        top_k: int = 10,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find visually similar products.
        
        Args:
            image: Query image
            top_k: Number of results
            exclude_ids: Product IDs to exclude
        
        Returns:
            List of similar products
        """
        if self._feature_index is None:
            raise ValueError("Similarity index not built. Call build_similarity_index first.")
        
        # Extract features for query
        features = self.extract_features([image])[0]
        
        # Compute similarities
        similarities = np.dot(self._feature_index, features)
        
        # Sort
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            product_id = self._indexed_ids[idx]
            
            if exclude_ids and product_id in exclude_ids:
                continue
            
            results.append({
                "product_id": product_id,
                "similarity": float(similarities[idx]),
                "rank": len(results) + 1
            })
            
            if len(results) >= top_k:
                break
        
        return results


class ImageQualityNetwork(nn.Module):
    """
    Network for assessing product image quality.
    
    Predicts multiple quality aspects:
    - Technical quality (blur, noise, exposure)
    - Composition quality
    - Background quality
    - Overall usability score
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        base_channels: int = 32
    ):
        super().__init__()
        
        # Feature extractor (lightweight)
        self.features = nn.Sequential(
            ConvBlock(3, base_channels, 7, stride=2, padding=3),
            nn.MaxPool2d(2),
            
            ConvBlock(base_channels, base_channels * 2, 3, stride=2),
            ResidualBlock(base_channels * 2),
            
            ConvBlock(base_channels * 2, base_channels * 4, 3, stride=2),
            ResidualBlock(base_channels * 4),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Quality heads
        self.shared = nn.Sequential(
            nn.Linear(base_channels * 4, embedding_dim),
            nn.GELU(),
        )
        
        # Technical quality (blur, noise, exposure)
        self.technical_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )
        
        # Composition score
        self.composition_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        # Background quality
        self.background_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        # Overall usability
        self.usability_head = nn.Sequential(
            nn.Linear(embedding_dim + 5, 32),  # Include other scores
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        shared = self.shared(features)
        
        # Individual quality aspects
        technical = self.technical_head(shared)  # [blur, noise, exposure]
        composition = self.composition_head(shared)
        background = self.background_head(shared)
        
        # Combine for overall
        all_scores = torch.cat([technical, composition, background], dim=-1)
        usability_input = torch.cat([shared, all_scores], dim=-1)
        usability = self.usability_head(usability_input)
        
        return {
            "blur_score": technical[:, 0:1],
            "noise_score": technical[:, 1:2],
            "exposure_score": technical[:, 2:3],
            "composition_score": composition,
            "background_score": background,
            "usability_score": usability,
        }


class ImageQualityAssessor(BaseNeuralNetwork):
    """
    Product image quality assessment.
    """
    
    MODEL_TYPE = "vision"
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str = "quality_assessor",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "embedding_dim": 128,
            "base_channels": 32,
            "input_size": 224,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 30,
        }
        if config:
            default_config.update(config)
        
        super().__init__(model_name, version, default_config)
    
    def build_model(self) -> nn.Module:
        return ImageQualityNetwork(
            embedding_dim=self.config["embedding_dim"],
            base_channels=self.config["base_channels"],
        )
    
    def get_loss_function(self) -> nn.Module:
        return nn.MSELoss()
    
    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality.
        
        Args:
            image: Input image
        
        Returns:
            Quality scores
        """
        self.model.eval()
        
        # Preprocess
        from PIL import Image
        import torchvision.transforms as T
        
        input_size = self.config["input_size"]
        
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
        
        return {
            "blur_score": outputs["blur_score"].item(),
            "noise_score": outputs["noise_score"].item(),
            "exposure_score": outputs["exposure_score"].item(),
            "composition_score": outputs["composition_score"].item(),
            "background_score": outputs["background_score"].item(),
            "usability_score": outputs["usability_score"].item(),
            "is_acceptable": outputs["usability_score"].item() > 0.6,
        }
    
    def batch_assess(
        self,
        images: List[np.ndarray],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """Assess quality for multiple images."""
        from PIL import Image
        import torchvision.transforms as T
        
        self.model.eval()
        
        input_size = self.config["input_size"]
        
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            tensors = []
            
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                tensors.append(transform(img))
            
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            for j in range(len(batch_images)):
                all_results.append({
                    "blur_score": outputs["blur_score"][j].item(),
                    "noise_score": outputs["noise_score"][j].item(),
                    "exposure_score": outputs["exposure_score"][j].item(),
                    "composition_score": outputs["composition_score"][j].item(),
                    "background_score": outputs["background_score"][j].item(),
                    "usability_score": outputs["usability_score"][j].item(),
                    "is_acceptable": outputs["usability_score"][j].item() > 0.6,
                })
        
        return all_results
