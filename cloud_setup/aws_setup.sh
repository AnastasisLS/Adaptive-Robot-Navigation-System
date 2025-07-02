#!/bin/bash

# AWS EC2 Setup Script for Adaptive Robot Navigation System
# This script sets up an EC2 instance to run your experiments

echo "Setting up AWS EC2 instance for experiments..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3 python3-pip python3-venv git

# Install system dependencies for OpenCV
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib seaborn pyyaml gym stable-baselines3 opencv-python tqdm pandas scikit-learn pytest

# Clone or copy your project (adjust the path)
# Option 1: If you uploaded to S3
# aws s3 cp s3://your-bucket/project.zip .
# unzip project.zip

# Option 2: If you're using git
# git clone https://github.com/yourusername/adaptive-robot-navigation.git

# Install the project
cd "Project 1 - Adaptive Robot Navigation System"
pip install -e .

# Create a script to run experiments
cat > run_experiments.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set display for matplotlib (headless mode)
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Run experiments
echo "Starting experiments..."

# Basic navigation experiment
python examples/basic_navigation.py --episodes 500

# Comparison experiment
python examples/comparison_experiment.py --episodes 100

# Save results to S3 (optional)
# aws s3 cp data/experiments/ s3://your-bucket/results/ --recursive

echo "Experiments completed!"
EOF

chmod +x run_experiments.sh

echo "Setup complete! Run './run_experiments.sh' to start experiments." 