#!/bin/bash

# Environment name from environment.yml
ENV_NAME="pytorch-transfer-learning-debugger"

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Found existing environment '$ENV_NAME'. Removing..."
    conda deactivate
    conda env remove --name $ENV_NAME
fi

# Create the Conda environment
echo "Creating new environment '$ENV_NAME'..."
conda env create -f environment.yml

# Activate the environment
conda activate $ENV_NAME

# Install the package in editable mode
pip install -e .

echo "Environment setup complete! Activate it with: conda activate $ENV_NAME"


# run file ----------------------------------------------------------------------------->>
# chmod +x setup_env.sh ## grant terminal access
# ./setup_env.sh

# required to build package
# pip install build twine

# run manually
# conda env create -f environment.yml
# conda activate pytorch-debug

# activate
# conda activate pytorch-transfer-learning-debugger
# conda activate /Users/zachwolpe/miniforge3/envs/CDB/envs/pytorch-transfer-learning-debugger
# # run file ----------------------------------------------------------------------------->>
