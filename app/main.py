import click

@click.group()
def cli():
    """Audora Audio Processor CLI"""
    pass

@cli.command()
def worker():
    """Start Celery worker"""
    import subprocess
    subprocess.call(["celery", "-A", "app.tasks", "worker", "--loglevel=info"])

@cli.command()
def beat():
    """Start Celery beat scheduler"""
    import subprocess
    subprocess.call(["celery", "-A", "app.tasks", "beat", "--loglevel=info"])
