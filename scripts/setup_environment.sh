#!/bin/bash

# Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda init bash
else
    echo "Miniconda already installed"
fi

# Create and activate Conda environment
conda create -n yolo_env python=3.12 -y
source $HOME/miniconda3/etc/profile.d/conda.sh

# Install dependencies
pip install ultralytics torch torchvision numpy pandas tqdm joblib scipy matplotlib shapely scikit-learn opencv-python tables pyproj
echo "Environment setup complete. Activate with: conda activate yolo_env"
