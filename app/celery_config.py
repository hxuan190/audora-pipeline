"""
Celery configuration for Audora audio processing
"""

import os
from kombu import Queue

# Redis configuration
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Celery settings
broker_url = redis_url
result_backend = redis_url

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True

# Task routing and queues
task_routes = {
    'app.tasks.validate_audio_upload': {'queue': 'validation'},
    'app.tasks.process_master_audio': {'queue': 'processing'},
    'app.tasks.generate_streaming_formats': {'queue': 'formats'},
    'app.tasks.finalize_track_processing': {'queue': 'finalization'},
    'app.tasks.generate_audio_preview': {'queue': 'previews'},
}

task_default_queue = 'default'
task_default_exchange = 'default'
task_default_exchange_type = 'direct'
task_default_routing_key = 'default'

task_queues = (
    Queue('validation', routing_key='validation'),
    Queue('processing', routing_key='processing'),
    Queue('formats', routing_key='formats'),
    Queue('finalization', routing_key='finalization'),
    Queue('previews', routing_key='previews'),
    Queue('default', routing_key='default'),
)

# Worker settings
worker_prefetch_multiplier = 1  # Process one task at a time for heavy audio processing
worker_max_tasks_per_child = 10  # Restart workers to prevent memory leaks
worker_disable_rate_limits = True
worker_send_task_events = True

# Task execution settings
task_acks_late = True  # Acknowledge task only after completion
task_reject_on_worker_lost = True
task_soft_time_limit = 300  # 5 minutes soft limit
task_time_limit = 600  # 10 minutes hard limit

# Results settings
result_expires = 3600  # Keep results for 1 hour
result_persistent = True

# Monitoring
worker_send_task_events = True
task_send_sent_event = True

# Beat schedule for periodic tasks
beat_schedule = {
    'cleanup-temp-files': {
        'task': 'app.tasks.cleanup_temp_files',
        'schedule': 300.0,  # Every 5 minutes
    },
    'monitor-processing-queue': {
        'task': 'app.tasks.monitor_processing_status',
        'schedule': 60.0,  # Every minute
    },
}

# Security settings
worker_hijack_root_logger = False
worker_log_color = False

# Performance settings
broker_connection_retry_on_startup = True
broker_connection_retry = True
broker_connection_max_retries = 10