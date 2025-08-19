"""
Audora Audio Processor CLI
"""

import click
import os
from app.tasks import queue_audio_processing, get_processing_status

@click.group()
def cli():
    """Audora Audio Processor CLI"""
    pass

@cli.command()
@click.option('--concurrency', '-c', default=2, help='Number of worker processes')
@click.option('--queue', '-q', default='default', help='Queue to process')
@click.option('--loglevel', '-l', default='info', help='Log level')
def worker(concurrency, queue, loglevel):
    """Start Celery worker"""
    import subprocess
    cmd = [
        "celery", "-A", "app.celery_app", "worker",
        f"--concurrency={concurrency}",
        f"--queues={queue}",
        f"--loglevel={loglevel}",
        "--task-events"
    ]
    subprocess.call(cmd)

@cli.command()
@click.option('--loglevel', '-l', default='info', help='Log level')
def beat(loglevel):
    """Start Celery beat scheduler"""
    import subprocess
    subprocess.call([
        "celery", "-A", "app.celery_app", "beat", 
        f"--loglevel={loglevel}"
    ])

@cli.command()
@click.option('--port', '-p', default=5555, help='Monitor port')
def monitor():
    """Start Flower monitoring"""
    import subprocess
    subprocess.call([
        "celery", "-A", "app.celery_app", "flower",
        f"--port={port}"
    ])

@cli.command()
@click.argument('file_path')
@click.argument('artist_id')
@click.option('--title', default='Test Track', help='Track title')
@click.option('--genre', default='Electronic', help='Track genre')
def process_file(file_path, artist_id, title, genre):
    """Process a local audio file (for testing)"""
    if not os.path.exists(file_path):
        click.echo(f"Error: File {file_path} not found")
        return
    
    # In a real implementation, you'd upload to MinIO first
    # For testing, we'll simulate with the file path as key
    metadata = {
        'title': title,
        'genre': genre,
        'artist_id': artist_id
    }
    
    task_id = queue_audio_processing(
        file_key=file_path,  # This would be MinIO key in production
        artist_id=artist_id,
        metadata=metadata
    )
    
    click.echo(f"Queued processing with task ID: {task_id}")
    click.echo(f"Monitor progress with: audora-processor status {task_id}")

@cli.command()
@click.argument('task_id')
def status(task_id):
    """Check processing status"""
    result = get_processing_status(task_id)
    
    click.echo(f"Task ID: {result['task_id']}")
    click.echo(f"Status: {result['status']}")
    
    if result['result']:
        click.echo(f"Result: {result['result']}")
    
    if result['traceback']:
        click.echo(f"Error: {result['traceback']}")

@cli.command()
def queues():
    """Show queue status"""
    import subprocess
    subprocess.call([
        "celery", "-A", "app.celery_app", "inspect", "active_queues"
    ])

@cli.command()
def purge():
    """Purge all queues (use with caution!)"""
    import subprocess
    if click.confirm('This will delete all pending tasks. Are you sure?'):
        subprocess.call([
            "celery", "-A", "app.celery_app", "purge", "-f"
        ])

if __name__ == '__main__':
    cli()