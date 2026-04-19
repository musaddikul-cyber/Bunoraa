"""
Multi-Redis Celery Workers Configuration

Workers are configured per Redis backend:
- Upstash (200MB): ML workloads, heavy computation
- Aiven (1GB): General Celery tasks (default), API caching, channels
- Render (25MB): Simple tasks, notifications, chat

Usage:
    # Default worker (Aiven)
    celery -A core worker -Q celery,payments,notifications,mails -l info
    
    # ML Worker (Upstash)
    celery -A core worker -Q ml,compute -l info --hostname=ml_worker@%h
    
    # Simple Task Worker (Render)
    celery -A core worker -Q simple,chat,realtime -l info --hostname=simple_worker@%h
"""
import os
from celery import Celery
from kombu import Queue, Exchange


def configure_multi_redis_queues(app: Celery):
    """Configure task routing to different Redis backends."""
    
    # Define exchanges
    default_exchange = Exchange('celery', type='direct')
    simple_exchange = Exchange('simple', type='direct')
    ml_exchange = Exchange('ml', type='direct')
    compute_exchange = Exchange('compute', type='direct')
    priority_exchange = Exchange('priority', type='direct')
    
    # Define queues with their respective backends
    app.conf.task_queues = (
        # Default queue -> Aiven (1GB)
        Queue('celery', default_exchange, routing_key='celery'),
        Queue('payments', default_exchange, routing_key='payments'),
        Queue('mails', default_exchange, routing_key='mails'),
        
        # Simple tasks queue -> Render (25MB)
        Queue('simple', simple_exchange, routing_key='simple'),
        Queue('notifications', simple_exchange, routing_key='notifications'),
        Queue('chat', simple_exchange, routing_key='chat'),
        Queue('realtime', simple_exchange, routing_key='realtime'),
        
        # ML queue -> Upstash (200MB)
        Queue('ml', ml_exchange, routing_key='ml'),
        Queue('compute', compute_exchange, routing_key='compute'),
        Queue('training', compute_exchange, routing_key='training'),
        Queue('analytics', compute_exchange, routing_key='analytics'),
        
        # Priority queue -> Upstash (fast responses)
        Queue('priority', priority_exchange, routing_key='priority'),
    )
    
    app.conf.task_default_queue = 'celery'
    app.conf.task_default_exchange = 'celery'
    app.conf.task_default_routing_key = 'celery'


def get_worker_concurrency(queue_type: str) -> int:
    """Get appropriate worker concurrency based on queue type and Redis limits."""
    
    concurrency_map = {
        'aiven': 2,      # 200MB - conservative
        'upstash': 4,        # 1GB - can handle more
        'render': 1,       # 25MB - minimal
    }
    
    return concurrency_map.get(queue_type, 2)


# Worker startup commands for different Redis backends
WORKER_COMMANDS = {
    'aiven': (
        'celery -A core worker '
        '-n aiven_worker@%h '
        '--pool=solo '
        '-Q celery,payments,mails,priority '
        '-l info '
        '--max-tasks-per-child=50 '
        '--optimize'
    ),
    'upstash': (
        'celery -A core worker '
        '-n upstash_worker@%h '
        '--pool=prefork '
        '--concurrency=4 '
        '-Q ml,compute,training,analytics '
        '-l info '
        '--max-tasks-per-child=100 '
        '--optimize'
    ),
    'render': (
        'celery -A core worker '
        '-n render_worker@%h '
        '--pool=solo '
        '-Q simple,notifications,chat,realtime '
        '-l info '
        '--max-tasks-per-child=20 '
        '--optimize'
    ),
}


# Production Render deployment configuration
RENDER_WORKER_CONFIG = {
    'celery': {
        'type': 'background_worker',
        'name': 'Bunoraa-Celery-Primary',
        'startCommand': WORKER_COMMANDS['celery'],
        'env_vars': {
            'CELERY_BROKER_URL': '${CELERY_REDIS_URL}',
            'CELERY_RESULT_BACKEND': '${CELERY_REDIS_URL}',
        }
    },
    'upstash': {
        'type': 'background_worker', 
        'name': 'Bunoraa-Celery-Upstash-ML',
        'startCommand': WORKER_COMMANDS['upstash'],
        'env_vars': {
            'CELERY_BROKER_URL': '${ML_REDIS_URL}',
            'CELERY_RESULT_BACKEND': '${ML_REDIS_URL}',
            'ML_BROKER_URL': '${ML_REDIS_URL}',
            'ML_RESULT_BACKEND': '${ML_REDIS_URL}',
            'ML_ENABLED': 'true',
            'ML_CELERY_BEAT_ENABLED': 'true',
        }
    },
    'render': {
        'type': 'background_worker',
        'name': 'Bunoraa-Celery-Render-Simple',
        'startCommand': WORKER_COMMANDS['render'],
        'env_vars': {
            'CELERY_BROKER_URL': '${RENDER_REDIS_URL}',
            'CELERY_RESULT_BACKEND': '${RENDER_REDIS_URL}',
        }
    },
}
