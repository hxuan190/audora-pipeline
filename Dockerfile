# Dockerfile
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    libsndfile1-dev \
    libffi-dev \
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Create directory for temporary files
RUN mkdir -p /tmp/audora-processing

# Set environment variables
ENV PYTHONPATH=/app
ENV TEMP_DIR=/tmp/audora-processing

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import app; print('OK')" || exit 1

# Default command
CMD ["audora-processor", "worker"]