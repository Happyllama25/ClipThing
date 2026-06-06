# FROM python:3.11-slim
FROM python:alpine

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

RUN apk update && apk add --no-cache ffmpeg

WORKDIR /app

# Copy dependencies and install (excluding PyQt5 for container)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application files
COPY ClipThing.py .
COPY index.html .
COPY loading.jpg .

# Create directories for clips and data
RUN mkdir -p /app/clips /app/data

# Set environment variable for clips root
ENV CLIPS_ROOT=/app/clips

# Expose the API port
EXPOSE 8000

# Run with -nogui flag (no tray icon in container)
CMD ["python", "ClipThing.py", "-nogui"]
