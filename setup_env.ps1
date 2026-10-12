# =============================================================================
# Environment Setup Script for Computer Vision Specialist Track
# For Windows PowerShell
# =============================================================================

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " CV Specialist Track — Environment Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion"

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch
Write-Host ""
Write-Host "Installing PyTorch..." -ForegroundColor Yellow

# Check for NVIDIA GPU
$hasGpu = $false
try {
    $null = Get-Command nvidia-smi -ErrorAction Stop
    nvidia-smi
    $hasGpu = $true
} catch {
    $hasGpu = $false
}

if ($hasGpu) {
    Write-Host "NVIDIA GPU detected. Installing CUDA build..." -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "No NVIDIA GPU detected. Installing CPU build..." -ForegroundColor Yellow
    Write-Host "For GPU training, use Google Colab or Kaggle Notebooks."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Install shared dependencies
Write-Host ""
Write-Host "Installing shared dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install Week 1 dependencies
Write-Host ""
Write-Host "Installing Week 1 dependencies..." -ForegroundColor Yellow
pip install -r week1_foundations/requirements.txt

# Verify installation
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Verifying installation..." -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

python -c @"
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
print('All core packages installed successfully!')
"@

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To set up W&B (experiment tracking):" -ForegroundColor Yellow
Write-Host "  wandb login"
Write-Host ""
Write-Host "Start with Day 1:" -ForegroundColor Yellow
Write-Host "  cd week1_foundations\day1_pytorch_basics"
Write-Host "  python day1_pytorch_basics.py"
