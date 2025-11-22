#!/bin/bash

# Define environment name
ENV_NAME="crypto_research_env"

echo "=================================================="
echo "Setting up Conda Environment: $ENV_NAME"
echo "=================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi

# Create environment with Python 3.11.8
echo "Creating environment with Python 3.11.8..."
conda create -n $ENV_NAME python=3.11.8 -y

# Activate environment (this might not work in subshell, so we use conda run later)
# But for user convenience, we print instructions
echo ""
echo "Environment created."

# Install dependencies using pip inside the conda env
echo "Installing dependencies..."
# We use the full path to pip in the new env to ensure we install there
# Or use conda run
conda run -n $ENV_NAME pip install --upgrade pip
conda run -n $ENV_NAME pip install pandas numpy matplotlib seaborn scipy
conda run -n $ENV_NAME pip install scikit-learn pycaret lightgbm xgboost
conda run -n $ENV_NAME pip install ta arcticdb numba jupyter ipykernel

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run Jupyter Notebook:"
echo "  conda activate $ENV_NAME"
echo "  jupyter notebook"
echo "=================================================="
