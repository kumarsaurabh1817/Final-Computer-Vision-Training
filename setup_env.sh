#!/bin/bash
# =============================================================================
# Environment Setup Script for Computer Vision Specialist Track
# For Ubuntu 24.04 on WSL2
# =============================================================================
set -e

echo "============================================="
echo " CV Specialist Track — Environment Setup"
echo "============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.4 — adjust if needed)
echo ""
echo "Installing PyTorch..."
# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA build..."
    nvidia-smi
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "No NVIDIA GPU detected. Installing CPU build..."
    echo "For GPU training, use Google Colab or Kaggle Notebooks."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install shared dependencies
echo ""
echo "Installing shared dependencies..."
pip install -r requirements.txt

# Install Week 1 dependencies
echo ""
echo "Installing Week 1 dependencies..."
pip install -r week1_foundations/requirements.txt

# Verify installation
echo ""
echo "============================================="
echo " Verifying installation..."
echo "============================================="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:    {torch.version.cuda}')
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
import torchvision
print(f'torchvision:     {torchvision.__version__}')
import cv2
print(f'OpenCV:          {cv2.__version__}')
import numpy
print(f'NumPy:           {numpy.__version__}')
print()
print('✅ All core packages installed successfully!')
"

echo ""
echo "============================================="
echo " Setup complete!"
echo "============================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To set up W&B (experiment tracking):"
echo "  wandb login"
echo ""
echo "Start with Day 1:"
echo "  cd week1_foundations/day1_pytorch_basics"
echo "  python day1_pytorch_basics.py"
