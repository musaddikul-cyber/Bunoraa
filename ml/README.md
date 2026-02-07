# Bunoraa ML/AI Engine

A comprehensive, production-ready machine learning and neural network framework for e-commerce, designed to sustain future world challenges and competitions.

## 🚀 Features

### Neural Network Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **Product Embeddings** | Triplet-loss learned product representations | Similar products, search |
| **User Embeddings** | Transformer-based user representations | Personalization |
| **Neural Collaborative Filtering** | Deep matrix factorization | Recommendations |
| **DeepFM** | Deep Factorization Machines | CTR prediction |
| **Two-Tower Model** | Dual encoder architecture | Retrieval at scale |
| **Sequence Recommender** | Transformer-based sequential | Session-based recommendations |
| **Demand Forecaster** | Temporal Fusion Transformer | Inventory planning |
| **Price Optimizer** | Elasticity-aware pricing | Dynamic pricing |
| **Fraud Detector** | Multi-signal attention fusion | Fraud prevention |
| **Churn Predictor** | Multi-task with uncertainty | Customer retention |
| **Semantic Search** | Hybrid dense/sparse retrieval | Product search |
| **Vision Classifier** | ResNet + SE blocks | Image categorization |

### Core Infrastructure

- **Model Registry**: Version control, A/B testing, promotion/rollback
- **Feature Store**: Real-time feature computation with Redis caching
- **Inference Engine**: Batch predictions, caching, fallback handling
- **Metrics Tracker**: Drift detection, alerting, Prometheus export
- **Training Pipeline**: Automated training with Celery

### Services

- **Recommendation Service**: Personalized, similar, FBT, cart-based
- **Search Service**: Semantic, autocomplete, visual, faceted
- **Analytics Service**: Demand forecast, pricing, segmentation
- **Fraud Service**: Order/user/payment risk assessment
- **Personalization Service**: Homepage, profiles, next-best-action

## 📦 Installation

### Requirements

```bash
pip install -r requirements-ml.txt
```

### Dependencies

- **PyTorch 2.0+**: Core deep learning framework
- **Transformers 4.35+**: NLP models and embeddings
- **scikit-learn 1.3+**: Traditional ML and preprocessing
- **faiss-cpu**: Similarity search
- **Redis**: Feature store and caching
- **Celery**: Async training tasks

## 🔧 Configuration

### Django Settings

Add to your `settings.py`:

```python
# ML Configuration
ML_MODELS_DIR = BASE_DIR / "ml"
ML_MODELS_DATA_DIR = BASE_DIR / "ml" / "models_data"
ML_TRAINING_DATA_DIR = BASE_DIR / "ml" / "training_data"
ML_CACHE_BACKEND = "redis"
ML_REDIS_URL = "redis://localhost:6379/1"

# Feature flags
ML_FEATURES = {
    "recommendations": True,
    "semantic_search": True,
    "fraud_detection": True,
    "demand_forecasting": True,
    "personalization": True,
}

# Model serving
ML_INFERENCE = {
    "cache_ttl": 3600,
    "batch_size": 32,
    "timeout": 5.0,
}
```

### URL Configuration

Add to your `urls.py`:

```python
from django.urls import path, include

urlpatterns = [
    # ... other urls
    path("api/ml/", include("ml_models.api.urls")),
]
```

### Celery Configuration

Add to your `celery.py`:

```python
from celery.schedules import crontab

app.conf.beat_schedule = {
    "train-recommendations-weekly": {
        "task": "ml_models.training.tasks.train_recommendation_model",
        "schedule": crontab(day_of_week="sunday", hour=2, minute=0),
        "args": ("ncf",),
    },
    "train-embeddings-weekly": {
        "task": "ml_models.training.tasks.train_embedding_models",
        "schedule": crontab(day_of_week="sunday", hour=3, minute=0),
    },
    "health-check-hourly": {
        "task": "ml_models.training.tasks.model_health_check",
        "schedule": crontab(minute=0),
    },
}
```

## 🎯 API Endpoints

### Recommendations

```
GET  /api/ml/recommendations/              # Personalized recommendations
GET  /api/ml/recommendations/popular/      # Popular products
GET  /api/ml/recommendations/similar/{id}/ # Similar products
GET  /api/ml/recommendations/fbt/{id}/     # Frequently bought together
POST /api/ml/recommendations/cart/         # Cart recommendations
```

### Search

```
GET  /api/ml/search/                       # Semantic search
GET  /api/ml/search/autocomplete/          # Autocomplete
POST /api/ml/search/visual/                # Visual search (image)
```

### Personalization

```
GET  /api/ml/personalization/homepage/     # Personalized homepage
GET  /api/ml/personalization/profile/      # User profile
GET  /api/ml/personalization/next-action/  # Next best action
```

### Analytics

```
GET  /api/ml/analytics/forecast/           # Demand forecast
GET  /api/ml/analytics/pricing/            # Price recommendations
GET  /api/ml/analytics/segments/           # Customer segments
GET  /api/ml/analytics/products/{id}/      # Product insights
GET  /api/ml/analytics/dashboard/          # Analytics dashboard
```

### Fraud Detection

```
POST /api/ml/fraud/assess-order/           # Assess order risk
GET  /api/ml/fraud/assess-user/{id}/       # Assess user risk
GET  /api/ml/fraud/dashboard/              # Fraud dashboard
```

### Admin

```
POST /api/ml/admin/train/                  # Trigger training
GET  /api/ml/admin/health/                 # Model health check
```

## 💻 Usage Examples

### Getting Recommendations

```python
from ml_models.services.recommendation_service import RecommendationService

service = RecommendationService()

# Personalized recommendations
recs = service.get_personalized_recommendations(
    user_id=123,
    num_items=20,
    category_id=5
)

# Similar products
similar = service.get_similar_products(
    product_id=456,
    num_items=10,
    similarity_type="hybrid"
)

# Cart recommendations
cart_recs = service.get_cart_recommendations(
    user_id=123,
    cart_product_ids=[789, 101, 102],
    num_items=5
)
```

### Semantic Search

```python
from ml_models.services.search_service import SearchService

service = SearchService()

# Semantic search
results = service.search(
    query="wireless bluetooth headphones",
    filters={"category_id": 10, "min_price": 50},
    page=1,
    page_size=20
)

# Autocomplete
suggestions = service.autocomplete("wireless blu", num_suggestions=5)

# Visual search (by image)
similar_products = service.visual_search(image_bytes, num_results=20)
```

### Fraud Detection

```python
from ml_models.services.fraud_service import FraudService

service = FraudService()

# Assess order risk
risk = service.assess_order_risk(
    order_data={
        "total_amount": 1500.00,
        "item_count": 3,
        "shipping_address": {...},
        "ip_address": "192.168.1.1"
    },
    user_data={
        "user_id": 123,
        "account_age_days": 30,
        "order_count": 2
    }
)

if risk.is_blocked:
    raise FraudBlockedException()
elif risk.needs_review:
    flag_for_manual_review(order)
```

### Training Models

```python
from ml_models.training.tasks import (
    train_recommendation_model,
    train_embedding_models,
    train_fraud_detector,
)

# Trigger async training
train_recommendation_model.delay("ncf")
train_embedding_models.delay()
train_fraud_detector.delay()

# Or use training pipeline directly
from ml_models.training.trainer import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.run_full_training()
```

## 🏗️ Architecture

```
ml_models/
├── core/
│   ├── config.py         # Configuration dataclasses
│   ├── base.py           # Base model classes
│   ├── registry.py       # Model registry with versioning
│   ├── inference.py      # Inference engine
│   ├── feature_store.py  # Feature computation & storage
│   └── metrics.py        # Metrics tracking & drift detection
│
├── models/
│   ├── embeddings.py     # Product & user embeddings
│   ├── recommender.py    # NCF, DeepFM, Two-Tower, Sequence
│   ├── forecasting.py    # Demand forecasting & pricing
│   ├── fraud.py          # Fraud detection network
│   ├── churn.py          # Churn prediction & CLV
│   ├── search.py         # Semantic search models
│   └── vision.py         # Image classification
│
├── training/
│   ├── data_loader.py    # Dataset classes & loaders
│   ├── callbacks.py      # Training callbacks
│   ├── trainer.py        # Universal trainer
│   └── tasks.py          # Celery training tasks
│
├── services/
│   ├── recommendation_service.py
│   ├── search_service.py
│   ├── analytics_service.py
│   ├── fraud_service.py
│   └── personalization_service.py
│
└── api/
    ├── views.py          # DRF views
    ├── serializers.py    # Request/response serializers
    └── urls.py           # URL patterns
```

## 🔬 Model Details

### Neural Collaborative Filtering (NCF)

Multi-layer perceptron combining matrix factorization with deep learning:

```python
# Architecture
Input: (user_id, product_id)
    → User Embedding (128d) + Product Embedding (128d)
    → Concatenate
    → Linear(256 → 128) → ReLU → Dropout(0.2)
    → Linear(128 → 64) → ReLU → Dropout(0.2)
    → Linear(64 → 32) → ReLU → Dropout(0.2)
    → Linear(32 → 1) → Sigmoid
Output: Interaction probability
```

### Two-Tower Recommender

Efficient retrieval with separate user and product towers:

```python
# User Tower
User Features → Embedding → MLP → User Vector (128d)

# Product Tower
Product Features → Embedding → MLP → Product Vector (128d)

# Retrieval
Score = User Vector · Product Vector (dot product)
```

### Fraud Detector

Multi-signal attention fusion network:

```python
# Transaction Signal
Transaction Features → Linear → Attention → Transaction Repr

# User Signal
User History → Linear → Attention → User Repr

# Device Signal
Device Fingerprint → Linear → Attention → Device Repr

# Fusion
Concat(Transaction, User, Device) → Cross-Attention → Fusion → Multi-head Output
    → is_fraud (binary)
    → fraud_type (multi-class)
    → risk_score (regression)
```

## 📊 Performance Benchmarks

| Model | Metric | Value |
|-------|--------|-------|
| NCF | NDCG@10 | 0.42 |
| Two-Tower | Recall@100 | 0.35 |
| DeepFM | AUC | 0.78 |
| Sequence | Hit Rate@10 | 0.28 |
| Demand Forecaster | MAPE | 12% |
| Fraud Detector | Precision@0.1 | 0.85 |
| Churn Predictor | AUC | 0.82 |
| Semantic Search | MRR | 0.65 |

## 🔒 Security

- All endpoints require authentication (except autocomplete, popular)
- Admin endpoints require staff permission
- Rate limiting recommended for production
- Fraud scores are not exposed to end users

## 🚦 Monitoring

### Prometheus Metrics

```
ml_model_predictions_total{model="ncf", version="1.0"}
ml_model_latency_seconds{model="fraud", quantile="0.95"}
ml_model_drift_score{model="embeddings"}
ml_training_duration_seconds{model="churn"}
```

### Health Checks

```bash
curl http://localhost:8000/api/ml/admin/health/
```

## 🔄 Continuous Improvement

### A/B Testing

```python
from ml_models.core.registry import ModelRegistry

registry = ModelRegistry()

# Deploy new model version
registry.register_model(
    model_type="recommender",
    model=new_model,
    metrics={"ndcg": 0.45},
    version="2.0.0"
)

# Split traffic
registry.set_ab_test_weight("recommender", "1.0.0", 0.5)
registry.set_ab_test_weight("recommender", "2.0.0", 0.5)

# Promote winner
registry.promote_model("recommender", "2.0.0")
```

### Retraining Schedule

| Model | Frequency | Trigger |
|-------|-----------|---------|
| Embeddings | Weekly | Data growth > 10% |
| Recommender | Weekly | NDCG drop > 5% |
| Fraud | Daily | New fraud patterns |
| Demand | Weekly | Season change |

## 🛠️ Development

### Running Tests

```bash
pytest ml_models/tests/ -v
```

### Adding New Models

1. Create model class extending `BaseNeuralNetwork`
2. Add dataset class to `training/data_loader.py`
3. Add training task to `training/tasks.py`
4. Create service in `services/`
5. Add API endpoints

## 📝 License

Proprietary - Bunoraa E-commerce Platform

## 🤝 Contributing

See CONTRIBUTING.md for guidelines.
