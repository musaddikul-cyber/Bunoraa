# ML System Integration Guide

This document provides comprehensive instructions for integrating the ML system into the Bunoraa Django application.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Data Collection](#data-collection)
5. [Management Commands](#management-commands)
6. [API Endpoints](#api-endpoints)
7. [Celery Tasks](#celery-tasks)
8. [Frontend Integration](#frontend-integration)
9. [Auto-Training](#auto-training)
10. [Production Deployment](#production-deployment)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Add to Installed Apps

```python
# settings.py
INSTALLED_APPS = [
    ...
    'ml',
    ...
]
```

### 2. Add Middleware

```python
# settings.py
MIDDLEWARE = [
    ...
    'ml.middleware.MLTrackingMiddleware',
    'ml.middleware.MLProductTrackingMiddleware',
    ...
]
```

### 3. Add URL Patterns

```python
# urls.py
from django.urls import path, include

urlpatterns = [
    ...
    path('api/ml/', include('ml.api.urls')),
    ...
]
```

### 4. Add Frontend Tracking

```html
<!-- base.html -->
<script src="{% static 'js/ml-tracking.js' %}" data-api-url="/api/ml/track/"></script>
```

### 5. Configure for Production

```python
# settings/production.py
PRODUCTION = True
ML_AUTO_TRAINING = True
```

---

## Installation

### Requirements

```txt
# requirements.txt
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
redis>=4.0.0
celery>=5.2.0
geoip2>=4.0.0  # Optional, for location tracking
user-agents>=2.2.0  # For device detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Redis Setup

```bash
# Install Redis (Ubuntu)
sudo apt install redis-server

# Or using Docker
docker run -d -p 6379:6379 redis:alpine
```

---

## Configuration

### Basic Settings

```python
# settings.py

# Enable ML features
ML_ENABLED = True

# Redis URL for ML data storage
REDIS_URL = 'redis://localhost:6379/0'

# Auto-training configuration
ML_AUTO_TRAINING = True
ML_AUTO_TRAINING_CONFIG = {
    'enabled': True,
    'min_interactions': 1000,
    'min_users': 100,
    'min_products': 50,
    'new_interactions_threshold': 5000,
    'max_days_between_training': 7,
    'training_hour': 2,  # 2 AM
}
```

### Production Settings

```python
# settings/production.py

PRODUCTION = True  # Required for auto-training

ML_AUTO_TRAINING = True
ML_AUTO_RETRAIN_ON_DRIFT = True

# More conservative thresholds for production
ML_AUTO_TRAINING_CONFIG = {
    'enabled': True,
    'min_interactions': 5000,
    'min_users': 500,
    'min_products': 100,
    'new_interactions_threshold': 10000,
    'max_days_between_training': 7,
    'enable_drift_detection': True,
    'drift_threshold': 0.15,
    'max_concurrent_trainings': 2,
}
```

### Data Collection Settings

```python
ML_DATA_COLLECTION = {
    'enabled': True,
    'batch_size': 100,
    'flush_interval': 300,  # 5 minutes
    'retention_days': 90,
    'anonymize_ip': False,  # Set True for GDPR
    'collect_location': True,
    'collect_device_info': True,
}
```

### Tracking Settings

```python
ML_TRACKING = {
    'track_page_views': True,
    'track_product_views': True,
    'exclude_paths': [
        '/admin/',
        '/api/ml/',
        '/static/',
        '/media/',
    ],
    'exclude_bots': True,
}
```

---

## Data Collection

### Automatic Collection

The system automatically collects:

- **Page Views**: URL, time on page, scroll depth, referrer
- **Product Views**: Product details, source page, position
- **User Interactions**: Clicks, form interactions, search queries
- **Cart Events**: Add, remove, update quantities
- **Checkout Events**: Each step of checkout process
- **Purchase Events**: Order details, amounts, items
- **User Profile**: Device, browser, location, preferences

### Manual Event Tracking

```python
# In your views
from ml_models.signals import emit_product_view, emit_cart_add

# Track product view
emit_product_view(
    request,
    product=product,
    source_page='category',
    position=1
)

# Track cart add
emit_cart_add(
    request,
    product=product,
    quantity=2,
    variant='blue-large'
)
```

### Custom Event Tracking

```python
from ml_models.data_collection.events import EventTracker, EventType

tracker = EventTracker()

# Track custom event
tracker.track_event(
    session_id='session-123',
    user_id=user.id,
    event_type=EventType.CUSTOM,
    data={
        'action': 'video_play',
        'video_id': '456',
        'duration': 120,
    }
)
```

---

## Management Commands

### ml_train - Train ML Models

```bash
# Train all models
python manage.py ml_train

# Train specific model
python manage.py ml_train --model ncf

# Train with custom epochs
python manage.py ml_train --model deepfm --epochs 100

# Train asynchronously (Celery)
python manage.py ml_train --async
```

### ml_collect - Collect Training Data

```bash
# Process queued data
python manage.py ml_collect --process

# Collect user profiles
python manage.py ml_collect --profiles

# Collect product features
python manage.py ml_collect --products

# Export training data to files
python manage.py ml_collect --export
```

### ml_evaluate - Evaluate Models

```bash
# Evaluate all models
python manage.py ml_evaluate

# Evaluate specific model
python manage.py ml_evaluate --model ncf

# Generate report
python manage.py ml_evaluate --report --output report.json
```

### ml_status - Check System Status

```bash
# Full status
python manage.py ml_status

# Model health only
python manage.py ml_status --health

# Data status only
python manage.py ml_status --data

# Redis queue status
python manage.py ml_status --queue

# JSON output
python manage.py ml_status --json
```

### ml_auto_train - Manage Auto-Training

```bash
# Run auto-training check
python manage.py ml_auto_train

# Check without triggering
python manage.py ml_auto_train --check

# Force train all models
python manage.py ml_auto_train --force

# Force train specific model
python manage.py ml_auto_train --model ncf

# Show configuration
python manage.py ml_auto_train --config

# Show schedule
python manage.py ml_auto_train --schedule

# Show history
python manage.py ml_auto_train --history 10
```

---

## API Endpoints

### Tracking Endpoint

```
POST /api/ml/track/

Request:
{
    "events": [
        {
            "event_type": "page_view",
            "timestamp": "2024-01-15T10:30:00Z",
            "session_id": "abc123",
            "page_url": "/products/shoe-123"
        }
    ],
    "meta": {
        "batch_id": "batch-456"
    }
}

Response:
{
    "status": "ok",
    "processed": 1
}
```

### Recommendations Endpoint

```
GET /api/ml/recommendations/?type=personalized&limit=10

Response:
{
    "status": "ok",
    "recommendations": [
        {"product_id": "123", "score": 0.95},
        {"product_id": "456", "score": 0.87}
    ]
}
```

### Search Endpoint

```
GET /api/ml/search/?q=blue+shoes&limit=20

Response:
{
    "status": "ok",
    "query": "blue shoes",
    "results": [...],
    "total": 20
}
```

### Predictions Endpoint

```
POST /api/ml/predict/

Request:
{
    "type": "demand",
    "product_id": "123",
    "horizon": 30
}

Response:
{
    "status": "ok",
    "type": "demand",
    "prediction": {
        "forecast": [100, 120, 115, ...],
        "confidence": 0.85
    }
}
```

---

## Celery Tasks

### Configuration

```python
# celery.py
from celery import Celery

app = Celery('bunoraa')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks(['ml_models'])
```

### Add Beat Schedule

```python
# settings.py
from ml_models.tasks import get_celery_beat_schedule

CELERY_BEAT_SCHEDULE = {
    # Your existing tasks...
}
CELERY_BEAT_SCHEDULE.update(get_celery_beat_schedule())
```

### Available Tasks

| Task | Schedule | Description |
|------|----------|-------------|
| `ml.process_training_data` | Every 5 min | Process queued interactions |
| `ml.auto_training_check` | Every hour | Check if models need training |
| `ml.check_drift` | Daily 1 AM | Check for data drift |
| `ml.evaluate_models` | Daily 2 AM | Evaluate model performance |
| `ml.export_training_data` | Daily 3 AM | Export training data to files |
| `ml.generate_report` | Daily 6 AM | Generate daily ML report |
| `ml.cleanup_old_data` | Weekly Sun 4 AM | Clean up old data |
| `ml.scheduled_training` | Weekly Sat 2 AM | Scheduled model retraining |

### Manual Task Invocation

```python
from ml_models.tasks import train_model_task, process_training_data_task

# Train model asynchronously
result = train_model_task.delay('ncf', {}, 'manual')
print(f"Task ID: {result.id}")

# Check task status
print(f"Status: {result.status}")
print(f"Result: {result.result}")
```

---

## Frontend Integration

### Include Tracking Script

```html
<!-- base.html -->
{% load static %}

<!-- Add before </body> -->
<script src="{% static 'js/ml-tracking.js' %}" data-api-url="/api/ml/track/"></script>

<!-- Optional: Add user ID for personalization -->
{% if user.is_authenticated %}
<meta name="user-id" content="{{ user.id }}">
{% endif %}
```

### Track Product Views

```javascript
// When user views a product
BunoraaML.trackProductView({
    product_id: '123',
    product_name: 'Blue Running Shoes',
    category: 'Shoes',
    price: 99.99,
    discount: 10,
    is_new_arrival: true,
    is_bestseller: false,
    source_page: 'category',
    position: 5
});
```

### Track Cart Events

```javascript
// Add to cart
BunoraaML.trackAddToCart({
    product_id: '123',
    product_name: 'Blue Running Shoes',
    price: 99.99,
    quantity: 2,
    variant: 'blue',
    size: '42'
});

// Remove from cart
BunoraaML.trackRemoveFromCart({
    product_id: '123',
    quantity: 1
});
```

### Track Checkout

```javascript
// Each checkout step
BunoraaML.trackCheckout({
    step: 1,
    step_name: 'shipping',
    cart_value: 199.98,
    items_count: 2
});

// Purchase complete
BunoraaML.trackPurchase({
    order_id: 'ORD-12345',
    order_value: 199.98,
    items_count: 2,
    coupon: 'SAVE10',
    payment_method: 'credit_card'
});
```

### Track Search

```javascript
BunoraaML.trackSearch({
    query: 'blue shoes',
    results_count: 25,
    filters: { category: 'shoes', color: 'blue' },
    sort_by: 'relevance'
});
```

### Custom Events

```javascript
// Track any custom event
BunoraaML.track('video_play', {
    video_id: '456',
    duration: 120
});
```

---

## Auto-Training

### How It Works

1. **Data Collection**: User interactions are collected via middleware and signals
2. **Queue Processing**: Data is queued in Redis and processed in batches
3. **Threshold Checking**: System checks if retraining is needed based on:
   - Amount of new data
   - Time since last training
   - Performance drop detection
   - Data drift detection
4. **Training Trigger**: When conditions are met, training is triggered
5. **Model Update**: New models are validated and deployed

### Training Triggers

| Trigger | Priority | Description |
|---------|----------|-------------|
| New Model | 100 | Model has never been trained |
| Drift Detected | 90 | Data distribution has changed |
| Performance Drop | 85 | Model performance below threshold |
| Scheduled | 50 | Max days between training exceeded |
| Data Threshold | 40 | Sufficient new data collected |

### Monitoring Auto-Training

```bash
# Check current status
python manage.py ml_auto_train --check

# View training history
python manage.py ml_auto_train --history 20

# View configuration
python manage.py ml_auto_train --config
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Set `PRODUCTION = True`
- [ ] Configure Redis URL
- [ ] Set up Celery worker and beat
- [ ] Configure model storage directory
- [ ] Set up logging
- [ ] Initial model training completed
- [ ] API endpoints tested

### Docker Compose Example

```yaml
# docker-compose.yml
services:
  web:
    build: .
    environment:
      - PRODUCTION=true
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - celery_worker

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data

  celery_worker:
    build: .
    command: celery -A bunoraa worker -l info
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis

  celery_beat:
    build: .
    command: celery -A bunoraa beat -l info
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis

volumes:
  redis_data:
```

### Initial Training

```bash
# Collect existing data
python manage.py ml_collect --export

# Train all models
python manage.py ml_train --all

# Verify models
python manage.py ml_evaluate
```

### Monitoring

```bash
# Check system health
python manage.py ml_status

# View logs
tail -f logs/ml.log

# Check Celery tasks
celery -A bunoraa inspect active
```

---

## Troubleshooting

### Common Issues

#### Redis Connection Error

```
Error: Could not connect to Redis
```

Solution:
```bash
# Check Redis is running
redis-cli ping

# Check Redis URL in settings
REDIS_URL = 'redis://localhost:6379/0'
```

#### Models Not Loading

```
Error: Model not found: ncf
```

Solution:
```bash
# Train the model first
python manage.py ml_train --model ncf

# Check model files exist
ls ml_models/saved_models/
```

#### Training Data Insufficient

```
Warning: Insufficient data for training
```

Solution:
```bash
# Check data status
python manage.py ml_status --data

# Wait for more interactions or lower thresholds
```

#### Celery Tasks Not Running

```
No tasks are being processed
```

Solution:
```bash
# Start Celery worker
celery -A bunoraa worker -l info

# Start Celery beat (for scheduled tasks)
celery -A bunoraa beat -l info

# Check task is registered
celery -A bunoraa inspect registered
```

### Debug Mode

Enable detailed logging:

```python
# settings.py
LOGGING = {
    'loggers': {
        'bunoraa.ml': {
            'level': 'DEBUG',
        },
    },
}
```

### Getting Help

- Check [ml_models/settings.py](ml_models/settings.py) for configuration options
- Review [ml_models/auto_training.py](ml_models/auto_training.py) for auto-training logic
- Examine [ml_models/tasks.py](ml_models/tasks.py) for Celery task definitions

---

## File Structure

```
ml_models/
├── __init__.py
├── apps.py                 # Django app config
├── settings.py             # ML settings reference
├── middleware.py           # Tracking middleware
├── signals.py              # Django signals for tracking
├── api_views.py            # REST API views
├── api_urls.py             # API URL patterns
├── auto_training.py        # Auto-training system
├── tasks.py                # Celery tasks
├── services.py             # ML service layer
├── core/                   # Core ML infrastructure
│   ├── config.py           # Configuration
│   ├── data_loader.py      # Data loading
│   ├── metrics.py          # Metrics tracking
│   ├── registry.py         # Model registry
│   └── trainer.py          # Training pipeline
├── models/                 # Neural network models
│   ├── ncf.py              # Neural Collaborative Filtering
│   ├── deepfm.py           # DeepFM
│   ├── two_tower.py        # Two-Tower retrieval
│   ├── sequence.py         # Sequence models
│   └── ...
├── data_collection/        # Data collection module
│   ├── collector.py        # Main data collector
│   ├── events.py           # Event tracker
│   ├── user_profile.py     # User profile collector
│   └── product_analytics.py # Product analytics
├── management/
│   └── commands/           # Django management commands
│       ├── ml_train.py
│       ├── ml_collect.py
│       ├── ml_evaluate.py
│       ├── ml_status.py
│       └── ml_auto_train.py
└── saved_models/           # Trained model files
```
