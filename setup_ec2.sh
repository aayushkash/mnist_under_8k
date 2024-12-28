#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python dependencies
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv mnist_env
source mnist_env/bin/activate

# Install required packages
pip install torch torchvision
pip install jupyter matplotlib numpy tqdm
pip install torchsummary

# Setup Jupyter
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

# Create notebooks directory if it doesn't exist

echo "Setup complete! You can now run:"
echo "source mnist_env/bin/activate"
echo "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser" 