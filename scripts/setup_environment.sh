#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Data paths initialization
mkdir -p data/raw data/processed logs

# pre-commit hooks
pre-commit install

# Setup scripts
chmod +x scripts/*.sh

echo "Environment created successfully, activate it with: venv/bin/activate"