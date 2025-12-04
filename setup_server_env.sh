#!/bin/bash

# Stop on error
set -e

ENV_NAME="mouse_face"
PYTHON_VERSION="3.10"

echo "==================================================="
echo "Starting Server Environment Setup for $ENV_NAME"
echo "==================================================="

# 1. Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# 2. Create Conda Environment
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
# Remove if exists to start fresh (optional, prompt user?)
# conda env remove -n $ENV_NAME -y || true
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 3. Activate Environment
# Note: 'conda activate' might not work in shell scripts without init.
# We use 'source activate' or eval hook.
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 4. Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
# Installing for CUDA 12.1 (common for newer GPUs, compatible with 12.6 driver)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install other dependencies
echo "Installing other dependencies..."
pip install pandas matplotlib seaborn scikit-learn opencv-python-headless tqdm ipykernel lightning mlflow optuna

# 6. Install project in editable mode
echo "Installing project in editable mode..."
pip install -e .

echo "==================================================="
echo "Setup Complete!"
echo "To use the environment, run: conda activate $ENV_NAME"
echo "==================================================="
