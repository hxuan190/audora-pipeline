"""
Celery application instance for Audora
"""

from celery import Celery
from app.celery_config import *

def create_celery_app():
    """Create and configure Celery application"""
    app = Celery('audora-processor')
    
    # Load configuration
    app.config_from_object('app.celery_config')
    
    # Auto-discover tasks
    app.autodiscover_tasks(['app.tasks'])
    
    return app

# Create the main celery app instance
celery_app = create_celery_app()