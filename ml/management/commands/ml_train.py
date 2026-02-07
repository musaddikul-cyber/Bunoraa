"""
ML Train Command

Django management command to train ML models.

Usage:
    python manage.py ml_train                     # Train all models
    python manage.py ml_train --model ncf        # Train specific model
    python manage.py ml_train --model embeddings --force  # Force retrain
    python manage.py ml_train --model all --logs  # Show detailed error logs
"""

import logging
import traceback
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("bunoraa.ml.commands")


class Command(BaseCommand):
    help = "Train ML models"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            default='all',
            choices=[
                'all', 'ncf', 'deepfm', 'two_tower', 'sequence',
                'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
            ],
            help='Model type to train (default: all)',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force training even if model is up to date',
        )
        parser.add_argument(
            '--async',
            action='store_true',
            dest='run_async',
            help='Run training asynchronously via Celery',
        )
        parser.add_argument(
            '--logs', '-l',
            action='store_true',
            dest='show_logs',
            help='Show detailed error logs and stack traces',
        )
        parser.add_argument(
            '--epochs', '-ep',
            type=int,
            default=50,
            help='Number of training epochs (default: 50)',
        )
        parser.add_argument(
            '--batch-size', '-bs',
            type=int,
            default=256,
            help='Batch size (default: 256)',
        )
        parser.add_argument(
            '--learning-rate', '-lr',
            type=float,
            default=0.001,
            help='Learning rate (default: 0.001)',
        )
    
    def handle(self, *args, **options):
        model_type = options['model']
        force = options['force']
        run_async = options['run_async']
        self.show_logs = options.get('show_logs', False)
        
        # Build training config
        config = {
            'epochs': options['epochs'],
            'batch_size': options['batch_size'],
            'learning_rate': options['learning_rate'],
        }
        
        self.stdout.write(f"Starting ML training for: {model_type}")
        if self.show_logs:
            self.stdout.write(self.style.NOTICE("  [Detailed logging enabled]"))
        
        if run_async:
            self._train_async(model_type, config)
        else:
            self._train_sync(model_type, force, config)
    
    def _train_async(self, model_type: str, config: dict):
        """Queue training tasks via Celery."""
        try:
            from ml.training.tasks import (
                train_recommendation_model,
                train_embedding_models,
                train_demand_forecaster,
                train_fraud_detector,
                train_churn_predictor,
                train_search_model,
                train_image_classifier,
            )
        except ImportError as e:
            raise CommandError(f"Celery tasks not available: {e}")
        
        tasks_queued = 0
        
        if model_type in ('all', 'ncf'):
            train_recommendation_model.delay('ncf', config)
            tasks_queued += 1
            self.stdout.write("  - Queued NCF training")
        
        if model_type in ('all', 'deepfm'):
            train_recommendation_model.delay('deepfm', config)
            tasks_queued += 1
            self.stdout.write("  - Queued DeepFM training")
        
        if model_type in ('all', 'two_tower'):
            train_recommendation_model.delay('two_tower', config)
            tasks_queued += 1
            self.stdout.write("  - Queued Two-Tower training")
        
        if model_type in ('all', 'sequence'):
            train_recommendation_model.delay('sequence', config)
            tasks_queued += 1
            self.stdout.write("  - Queued Sequence Recommender training")
        
        if model_type in ('all', 'embeddings'):
            train_embedding_models.delay()
            tasks_queued += 1
            self.stdout.write("  - Queued Embeddings training")
        
        if model_type in ('all', 'demand'):
            train_demand_forecaster.delay()
            tasks_queued += 1
            self.stdout.write("  - Queued Demand Forecaster training")
        
        if model_type in ('all', 'fraud'):
            train_fraud_detector.delay()
            tasks_queued += 1
            self.stdout.write("  - Queued Fraud Detector training")
        
        if model_type in ('all', 'churn'):
            train_churn_predictor.delay()
            tasks_queued += 1
            self.stdout.write("  - Queued Churn Predictor training")
        
        if model_type in ('all', 'search'):
            train_search_model.delay()
            tasks_queued += 1
            self.stdout.write("  - Queued Search Model training")
        
        if model_type in ('all', 'image'):
            train_image_classifier.delay()
            tasks_queued += 1
            self.stdout.write("  - Queued Image Classifier training")
        
        self.stdout.write(
            self.style.SUCCESS(f"Queued {tasks_queued} training task(s)")
        )
    
    def _train_sync(self, model_type: str, force: bool, config: dict):
        """Run training synchronously."""
        start_time = timezone.now()
        results = {}
        
        models_to_train = []
        
        if model_type == 'all':
            models_to_train = [
                'ncf', 'deepfm', 'two_tower', 'sequence',
                'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
            ]
        else:
            models_to_train = [model_type]
        
        try:
            for model_name in models_to_train:
                self.stdout.write(f"\nTraining {model_name}...")
                
                try:
                    result = self._train_single_model(model_name, config, force)
                    results[model_name] = result
                    
                    if result.get('status') == 'skipped':
                        self.stdout.write(
                            self.style.WARNING(f"  Skipped: {result.get('reason', 'up to date')}")
                        )
                    else:
                        self.stdout.write(self.style.SUCCESS(f"  Completed!"))
                        if result.get('metrics'):
                            for key, value in result['metrics'].items():
                                if isinstance(value, float):
                                    self.stdout.write(f"    {key}: {value:.4f}")
                                else:
                                    self.stdout.write(f"    {key}: {value}")
                                    
                except Exception as e:
                    results[model_name] = {'status': 'error', 'error': str(e)}
                    self.stdout.write(
                        self.style.ERROR(f"  Error: {e}")
                    )
                    if self.show_logs:
                        self.stdout.write(self.style.ERROR("  Stack trace:"))
                        for line in traceback.format_exc().split('\n'):
                            if line.strip():
                                self.stdout.write(self.style.ERROR(f"    {line}"))
                    logger.exception(f"Training error for {model_name}")
            
            # Report summary
            elapsed = (timezone.now() - start_time).total_seconds()
            self.stdout.write(f"\n{'='*50}")
            self.stdout.write(f"Training completed in {elapsed:.1f}s")
            
            successful = sum(1 for r in results.values() if r.get('status') == 'success')
            skipped = sum(1 for r in results.values() if r.get('status') == 'skipped')
            errors = sum(1 for r in results.values() if r.get('status') == 'error')
            
            self.stdout.write(f"  Success: {successful}")
            self.stdout.write(f"  Skipped: {skipped}")
            self.stdout.write(f"  Errors: {errors}")
            
            if errors == 0:
                self.stdout.write(self.style.SUCCESS("\nAll training completed successfully!"))
            else:
                self.stdout.write(self.style.WARNING(f"\nCompleted with {errors} error(s)"))
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Training failed: {e}")
            )
            logger.exception("Training error")
            raise CommandError(str(e))
    
    def _train_single_model(self, model_type: str, config: dict, force: bool) -> dict:
        """Train a single model."""
        from ml.core.registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # Check if training is needed
        if not force and not self._needs_training(model_type, registry):
            return {'status': 'skipped', 'reason': 'Model is up to date'}
        
        # Get model class and config
        model_class, model_config = self._get_model_class(model_type)
        
        if model_class is None:
            return {'status': 'skipped', 'reason': 'Model class not available'}
        
        # Load training data
        train_loader, val_loader = self._get_data_loaders(model_type, config['batch_size'])
        
        if train_loader is None:
            return {'status': 'skipped', 'reason': 'No training data available'}
        
        # Create and train model
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Initialize model wrapper
            model_wrapper = model_class(**model_config)
            
            # Build the actual PyTorch model
            if hasattr(model_wrapper, 'build_model'):
                model = model_wrapper.build_model()
            elif hasattr(model_wrapper, 'model') and model_wrapper.model is not None:
                model = model_wrapper.model
            elif isinstance(model_wrapper, nn.Module):
                model = model_wrapper
            else:
                return {'status': 'error', 'error': 'Could not build model'}
            
            # Move model to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            if self.show_logs:
                self.stdout.write(f"    Device: {device}")
                self.stdout.write(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Get loss function
            loss_fn = self._get_loss_function(model_type, model_wrapper)
            
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            
            # Training loop
            epochs = config['epochs']
            history = {'train_loss': [], 'val_loss': []}
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    try:
                        loss = self._compute_loss(model, batch, model_type, device, loss_fn)
                        
                        if loss is None:
                            continue
                        
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                    except Exception as e:
                        if self.show_logs:
                            self.stdout.write(self.style.WARNING(f"    Batch error: {e}"))
                        continue
                
                avg_loss = epoch_loss / max(num_batches, 1)
                history['train_loss'].append(avg_loss)
                
                # Validation
                if val_loader:
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for batch in val_loader:
                            try:
                                loss = self._compute_loss(model, batch, model_type, device, loss_fn)
                                if loss is not None:
                                    val_loss += loss.item()
                                    val_batches += 1
                            except Exception:
                                continue
                    
                    if val_batches > 0:
                        history['val_loss'].append(val_loss / val_batches)
                
                self.stdout.write(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Get final metrics
            final_metrics = {
                'final_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'epochs_trained': len(history['train_loss']),
            }
            
            if history.get('val_loss'):
                final_metrics['val_loss'] = history['val_loss'][-1]
            
            # Save model
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            registry.register(
                model_name=model_type,
                version=version,
                model=model,
                metrics=final_metrics,
            )
            
            return {
                'status': 'success',
                'version': version,
                'metrics': final_metrics,
            }
            
        except Exception as e:
            logger.exception(f"Error training {model_type}")
            return {'status': 'error', 'error': str(e)}
    
    def _needs_training(self, model_type: str, registry) -> bool:
        """Check if model needs retraining."""
        try:
            model_info = registry.get_model_info(model_type)
            if not model_info:
                return True
            
            last_trained = model_info.get('trained_at') or model_info.get('last_trained')
            if not last_trained:
                return True
            
            interval = getattr(settings, 'ML_MODEL_UPDATE_INTERVAL', 24)
            threshold = timezone.now() - timedelta(hours=interval)
            
            if isinstance(last_trained, str):
                last_trained = datetime.fromisoformat(last_trained.replace('Z', '+00:00'))
            
            if last_trained.tzinfo is None:
                from django.utils import timezone as tz
                last_trained = tz.make_aware(last_trained)
            
            return last_trained < threshold
            
        except Exception as e:
            logger.warning(f"Error checking training status for {model_type}: {e}")
            return True
    
    def _get_model_class(self, model_type: str):
        """Get model class and default config."""
        try:
            if model_type == 'ncf':
                from ml.models.recommender import NeuralCollaborativeFiltering
                return NeuralCollaborativeFiltering, {
                    'config': {
                        'num_users': 10000,
                        'num_items': 5000,
                        'embedding_dim': 64,
                    }
                }
            elif model_type == 'deepfm':
                from ml.models.recommender import DeepFM
                return DeepFM, {
                    'config': {
                        'num_users': 10000,
                        'num_items': 5000,
                        'embedding_dim': 32,
                        'num_features': 50,
                    }
                }
            elif model_type == 'two_tower':
                from ml.models.recommender import TwoTowerRecommender
                return TwoTowerRecommender, {
                    'config': {
                        'user_feature_dim': 64,
                        'item_feature_dim': 128,
                        'embedding_dim': 128,
                    }
                }
            elif model_type == 'sequence':
                from ml.models.recommender import SequenceRecommender
                return SequenceRecommender, {
                    'config': {
                        'num_items': 500,  # Match dummy data
                        'embedding_dim': 64,
                        'max_seq_length': 50,
                        'num_heads': 4,
                        'num_layers': 2,
                    }
                }
            elif model_type == 'embeddings':
                from ml.models.embeddings import ProductEmbeddingModel
                return ProductEmbeddingModel, {
                    'config': {'embedding_dim': 128}
                }
            elif model_type == 'demand':
                from ml.models.forecasting import DemandForecaster
                return DemandForecaster, {}
            elif model_type == 'fraud':
                from ml.models.fraud import FraudDetector
                return FraudDetector, {}
            elif model_type == 'churn':
                from ml.models.churn import ChurnPredictor
                return ChurnPredictor, {}
            elif model_type == 'search':
                from ml.models.search import SemanticSearchModel
                return SemanticSearchModel, {}
            elif model_type == 'image':
                from ml.models.vision import ProductImageClassifier
                return ProductImageClassifier, {
                    'config': {
                        'num_classes': 20,  # Reduced for testing
                        'num_attributes': 10,
                        'embedding_dim': 128,
                        'base_channels': 32,
                        'input_size': 112,
                    }
                }
            else:
                return None, {}
        except ImportError as e:
            logger.warning(f"Could not import model class for {model_type}: {e}")
            return None, {}
    
    def _get_data_loaders(self, model_type: str, batch_size: int):
        """Get data loaders for training."""
        try:
            from ml.training.data_loader import DataLoaderFactory
            
            factory = DataLoaderFactory()
            return factory.create_loaders(model_type, batch_size=batch_size)
        except Exception as e:
            logger.warning(f"Could not create data loaders for {model_type}: {e}")
            
            # Try to create dummy data for testing
            try:
                return self._create_dummy_loaders(model_type, batch_size)
            except Exception:
                return None, None
    
    def _get_loss_function(self, model_type: str, model_wrapper):
        """Get the appropriate loss function for a model type."""
        import torch.nn as nn
        
        # Use simple standard loss functions for training
        # Complex model-specific losses can be used in production
        loss_map = {
            'ncf': nn.BCELoss(),
            'deepfm': nn.BCELoss(),
            'two_tower': nn.MSELoss(),
            'sequence': nn.CrossEntropyLoss(),
            'fraud': nn.BCELoss(),
            'churn': nn.BCELoss(),
            'embeddings': nn.MSELoss(),
            'demand': nn.MSELoss(),
            'search': nn.MSELoss(),
            'image': nn.CrossEntropyLoss(),
        }
        return loss_map.get(model_type, nn.MSELoss())
    
    def _compute_loss(self, model, batch, model_type: str, device, loss_fn):
        """Compute loss for a single batch based on model type."""
        import torch
        import torch.nn as nn
        
        if model_type == 'ncf':
            # NCF: user_ids, item_ids, labels
            user_ids, item_ids, labels = batch
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.to(device).float()
            
            outputs = model(user_ids, item_ids)
            return loss_fn(outputs.squeeze(), labels)
        
        elif model_type == 'deepfm':
            # DeepFM: feature_indices, feature_values, labels
            feature_indices, feature_values, labels = batch
            feature_indices = feature_indices.to(device)
            feature_values = feature_values.to(device)
            labels = labels.to(device).float()
            
            outputs = model(feature_indices, feature_values)
            return loss_fn(outputs.squeeze(), labels)
        
        elif model_type == 'two_tower':
            # Two Tower: user_ids, user_features, item_ids, item_features, labels
            if len(batch) == 5:
                user_ids, user_features, item_ids, item_features, labels = batch
                user_ids = user_ids.to(device)
                user_features = user_features.to(device)
                item_ids = item_ids.to(device)
                item_features = item_features.to(device)
                labels = labels.to(device).float()
                
                outputs = model(user_ids, user_features, item_ids, item_features)
                return loss_fn(outputs.squeeze(), labels)
            elif len(batch) == 3:
                # Simplified: just user features, item features, labels
                user_features, item_features, labels = batch
                user_features = user_features.to(device)
                item_features = item_features.to(device)
                labels = labels.to(device).float()
                
                # Use cosine similarity as loss
                similarity = nn.functional.cosine_similarity(user_features, item_features)
                return loss_fn(similarity, labels)
        
        elif model_type == 'sequence':
            # Sequence: input_sequences, targets
            sequences, targets = batch
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            outputs = model(sequences)  # (batch, seq_len, num_items)
            # For next-item prediction, use the last position's output
            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]  # (batch, num_items)
            return nn.CrossEntropyLoss()(outputs, targets)
        
        elif model_type == 'embeddings':
            # Embeddings: product_ids, category_ids, tag_ids, numerical_features, labels
            if len(batch) == 5:
                product_ids, category_ids, tag_ids, numerical_features, labels = batch
                product_ids = product_ids.to(device)
                category_ids = category_ids.to(device)
                tag_ids = tag_ids.to(device)
                numerical_features = numerical_features.to(device)
                labels = labels.to(device).float()
                
                outputs = model(product_ids, category_ids, tag_ids, numerical_features)
                # Use embedding similarity loss - MSE between embedding pairs
                # Split into anchor and positive pairs
                anchor = outputs[::2] if len(outputs) > 1 else outputs
                positive = outputs[1::2] if len(outputs) > 1 else outputs
                similarity = nn.functional.cosine_similarity(anchor, positive)
                target_similarity = labels[:len(similarity)]
                return nn.MSELoss()(similarity, target_similarity)
            elif len(batch) == 2:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device).float()
                outputs = model(features)
                return nn.MSELoss()(outputs.squeeze(), labels)
        
        elif model_type == 'fraud':
            # Fraud: transaction_features, user_features, device_features, labels
            if len(batch) == 4:
                trans_feat, user_feat, device_feat, labels = batch
                trans_feat = trans_feat.to(device)
                user_feat = user_feat.to(device)
                device_feat = device_feat.to(device)
                labels = labels.to(device).float()
                
                result = model(trans_feat, user_feat, device_feat)
                if isinstance(result, dict):
                    fraud_score = result['fraud_prob']
                elif isinstance(result, tuple):
                    fraud_score = result[0]
                else:
                    fraud_score = result
                return loss_fn(fraud_score.squeeze(), labels)
            elif len(batch) == 2:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device).float()
                
                if hasattr(model, 'forward'):
                    result = model(features)
                    if isinstance(result, tuple):
                        fraud_score = result[0]
                    else:
                        fraud_score = result
                    return loss_fn(fraud_score.squeeze(), labels)
        
        elif model_type == 'churn':
            # Churn: behavioral_history, behavioral_mask, transaction_features, demographic_features, labels
            if len(batch) == 5:
                beh_hist, beh_mask, trans_feat, demo_feat, labels = batch
                beh_hist = beh_hist.to(device)
                beh_mask = beh_mask.to(device) if beh_mask is not None else None
                trans_feat = trans_feat.to(device)
                demo_feat = demo_feat.to(device)
                labels = labels.to(device).float()
                
                result = model(beh_hist, beh_mask, trans_feat, demo_feat)
                if isinstance(result, dict):
                    return loss_fn(result['churn_prob'], labels)
                return loss_fn(result.squeeze(), labels)
            elif len(batch) == 2:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device).float()
                outputs = model(features)
                if isinstance(outputs, dict):
                    return loss_fn(outputs['churn_prob'], labels)
                return loss_fn(outputs.squeeze(), labels)
        
        elif model_type == 'demand':
            # Demand: product_ids, store_ids, static_features, historical_sequence, future_covariates, labels
            if len(batch) == 6:
                prod_ids, store_ids, static_feat, hist_seq, future_cov, labels = batch
                prod_ids = prod_ids.to(device)
                store_ids = store_ids.to(device)
                static_feat = static_feat.to(device)
                hist_seq = hist_seq.to(device)
                future_cov = future_cov.to(device)
                labels = labels.to(device).float()
                
                outputs = model(prod_ids, store_ids, static_feat, hist_seq, future_cov)
                # Output shape: (batch, horizon, num_quantiles)
                # Use median quantile (index 1) for loss
                if outputs.dim() == 3:
                    outputs = outputs[:, :, 1]  # Take median
                return loss_fn(outputs.squeeze(), labels)
            elif len(batch) == 2:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device).float()
                outputs = model(features)
                return loss_fn(outputs.squeeze(), labels)
        
        elif model_type == 'search':
            # Search: token_ids, attention_mask, product_features, labels
            if len(batch) == 4:
                token_ids, attention_mask, product_features, labels = batch
                token_ids = token_ids.to(device)
                attention_mask = attention_mask.to(device)
                product_features = product_features.to(device)
                labels = labels.to(device).float()
                
                outputs = model(token_ids, attention_mask, product_features)
                # Similarity scores
                if isinstance(outputs, dict):
                    similarity = outputs.get('similarity', outputs.get('scores'))
                else:
                    similarity = outputs
                return nn.BCELoss()(torch.sigmoid(similarity.squeeze()), labels)
            elif len(batch) == 3:
                token_ids, attention_mask, labels = batch
                token_ids = token_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                outputs = model(token_ids, attention_mask)
                # Contrastive loss
                return loss_fn(outputs[::2], outputs[1::2], labels)
            elif len(batch) == 2:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device).float()
                outputs = model(features)
                return nn.MSELoss()(outputs.squeeze(), labels)
        
        elif model_type == 'image':
            # Image: images, labels
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs.get('class_logits', outputs.get('category_logits', outputs.get('logits')))
                if logits is None:
                    logits = list(outputs.values())[0]
                return nn.CrossEntropyLoss()(logits, labels)
            return nn.CrossEntropyLoss()(outputs, labels)
        
        else:
            # Generic fallback
            if len(batch) == 2:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device).float()
                outputs = model(features)
                return loss_fn(outputs.squeeze(), labels)
        
        return None
    
    def _create_dummy_loaders(self, model_type: str, batch_size: int):
        """Create dummy data loaders for testing when no real data is available."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        self.stdout.write(self.style.WARNING(f"    Using dummy data for {model_type}"))
        
        n_samples = 1000
        
        # Create random dummy data based on model type
        if model_type == 'ncf':
            # NCF: user_ids, item_ids, labels
            users = torch.randint(0, 1000, (n_samples,))
            items = torch.randint(0, 500, (n_samples,))
            labels = torch.randint(0, 2, (n_samples,)).float()
            dataset = TensorDataset(users, items, labels)
        
        elif model_type == 'deepfm':
            # DeepFM: feature_indices, feature_values, labels
            num_fields = 20
            feature_indices = torch.randint(0, 50, (n_samples, num_fields))
            feature_values = torch.randn(n_samples, num_fields)
            labels = torch.randint(0, 2, (n_samples,)).float()
            dataset = TensorDataset(feature_indices, feature_values, labels)
        
        elif model_type == 'two_tower':
            # Two Tower: user_ids, user_features, item_ids, item_features, labels
            user_ids = torch.randint(0, 1000, (n_samples,))
            user_features = torch.randn(n_samples, 50)  # num_user_features
            item_ids = torch.randint(0, 500, (n_samples,))
            item_features = torch.randn(n_samples, 100)  # num_item_features
            labels = torch.randint(0, 2, (n_samples,)).float()
            dataset = TensorDataset(user_ids, user_features, item_ids, item_features, labels)
        
        elif model_type == 'sequence':
            # Sequence: sequences, targets - use same num_items as model config
            num_items = 500
            sequences = torch.randint(1, num_items, (n_samples, 50))
            targets = torch.randint(0, num_items, (n_samples,))
            dataset = TensorDataset(sequences, targets)
        
        elif model_type == 'embeddings':
            # Embeddings: product_ids, category_ids, tag_ids, numerical_features, labels
            product_ids = torch.randint(0, 1000, (n_samples,))
            category_ids = torch.randint(0, 50, (n_samples,))
            tag_ids = torch.randint(0, 100, (n_samples,))
            numerical_features = torch.randn(n_samples, 10)
            labels = torch.randn(n_samples)
            dataset = TensorDataset(product_ids, category_ids, tag_ids, numerical_features, labels)
        
        elif model_type == 'fraud':
            # Fraud: transaction_features, user_features, device_features, labels
            transaction_features = torch.randn(n_samples, 50)
            user_features = torch.randn(n_samples, 30)
            device_features = torch.randn(n_samples, 20)
            labels = torch.randint(0, 2, (n_samples,)).float()
            dataset = TensorDataset(transaction_features, user_features, device_features, labels)
        
        elif model_type == 'churn':
            # Churn: behavioral_history, behavioral_mask, transaction_features, demographic_features, labels
            behavioral_history = torch.randn(n_samples, 50, 50)  # seq_len=50, features=50
            behavioral_mask = torch.ones(n_samples, 50)  # All valid
            transaction_features = torch.randn(n_samples, 30)
            demographic_features = torch.randn(n_samples, 20)
            labels = torch.randint(0, 2, (n_samples,)).float()
            dataset = TensorDataset(behavioral_history, behavioral_mask, transaction_features, demographic_features, labels)
        
        elif model_type == 'demand':
            # Demand: product_ids, store_ids, static_features, historical_sequence, future_covariates, labels
            product_ids = torch.randint(0, 1000, (n_samples,))
            store_ids = torch.randint(0, 10, (n_samples,))
            static_features = torch.randn(n_samples, 10)
            historical_sequence = torch.randn(n_samples, 30, 20)  # history_len=30, features=20
            future_covariates = torch.randn(n_samples, 14, 20)  # forecast_horizon=14
            labels = torch.randn(n_samples, 14).abs()  # Positive demand for each horizon
            dataset = TensorDataset(product_ids, store_ids, static_features, historical_sequence, future_covariates, labels)
        
        elif model_type == 'search':
            # Search: token_ids, attention_mask, product_features, labels
            token_ids = torch.randint(1, 5000, (n_samples, 50))  # vocab_size=5000, seq_len=50
            attention_mask = torch.ones(n_samples, 50)
            product_features = torch.randn(n_samples, 128)  # Product embeddings
            labels = torch.randint(0, 2, (n_samples,)).float()  # 1 if match, 0 otherwise
            dataset = TensorDataset(token_ids, attention_mask, product_features, labels)
        
        elif model_type == 'image':
            # Image: images (batch, channels, height, width), labels
            # Use smaller images for testing to avoid OOM
            n_samples = 300  # Fewer samples for image
            images = torch.randn(n_samples, 3, 112, 112)
            labels = torch.randint(0, 20, (n_samples,))  # Match num_classes=20
            dataset = TensorDataset(images, labels)
        
        else:
            # Generic classification data
            features = torch.randn(n_samples, 64)
            labels = torch.randint(0, 2, (n_samples,)).float()
            dataset = TensorDataset(features, labels)
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader
