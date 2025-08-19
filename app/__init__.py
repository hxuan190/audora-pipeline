"""
Audora Audio Processor Package
"""

from app.celery_app import celery_app
from app.tasks import (
    validate_audio_upload,
    process_master_audio,
    generate_streaming_formats,
    finalize_track_processing,
    generate_audio_preview,
    queue_audio_processing,
    get_processing_status
)

__version__ = "0.1.0"
__all__ = [
    'celery_app',
    'validate_audio_upload',
    'process_master_audio', 
    'generate_streaming_formats',
    'finalize_track_processing',
    'generate_audio_preview',
    'queue_audio_processing',
    'get_processing_status'
]
