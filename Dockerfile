# Dockerfile for Adaptive Robot Navigation System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the project in development mode
RUN pip install -e .

# Create directories for data
RUN mkdir -p data/experiments data/models

# Set environment variables for headless operation
ENV DISPLAY=:99
ENV PYTHONPATH=/app

# Create entry point script
RUN echo '#!/bin/bash\n\
# Start virtual display for matplotlib\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
\n\
# Run experiments based on arguments\n\
if [ "$1" = "basic" ]; then\n\
    python examples/basic_navigation.py --episodes ${2:-500}\n\
elif [ "$1" = "comparison" ]; then\n\
    python examples/comparison_experiment.py --episodes ${2:-100}\n\
else\n\
    echo "Usage: docker run <image> [basic|comparison] [episodes]"\n\
    echo "Example: docker run <image> basic 500"\n\
fi\n\
\n\
# Copy results to mounted volume if available\n\
if [ -d "/results" ]; then\n\
    cp -r data/experiments/* /results/\n\
fi' > /app/run_experiments.sh

RUN chmod +x /app/run_experiments.sh

# Set the entry point
ENTRYPOINT ["/app/run_experiments.sh"] 