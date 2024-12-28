# MNIST CNN Classifier

[![Python Tests](https://github.com/aayushkash/mnist_under_8k/actions/workflows/python-app.yml/badge.svg)](https://github.com/aayushkash/mnist_under_8k/actions/workflows/python-app.yml)

This project implements a Convolutional Neural Network (CNN) for classifying MNIST digits using PyTorch. The model is designed to be lightweight with less than 8000 parameters while maintaining good accuracy.

## Features

- Lightweight CNN architecture (<8K parameters)
- Batch Normalization for better training stability
- Dropout for regularization
- Global Average Pooling to reduce parameters
- Comprehensive logging system
- Unit tests compatible with GitHub Actions

## Project Structure
- mnist_cnn/
- notebook/
- |-ec2_training_final.ipynb
- |-final_model_collab.ipynb
- ├── src/
- │ ├── models
- │ │ ├── model_relu.py
- │ │ ├── model_batchnorm.py
- │ │ ├── model_dropout.py
- │ │ ├── model_gap.py
- │ ├── train.py # Training logic
- │ ├── data.py # Data loading utilities
- │ └── utils.py # Helper functions
- ├── tests/ # Unit tests
- ├── logs/ # Training logs
- └── requirements.txt

### Key Results
- Reaches 99.4% validation accuracy in just 13 epochs in M1/M2 Mac using MPS acceleration
- In Colab same model and seed it takes it reaches 99.39% Have to figure out why. Will Apply session 7 methods to imporove on collab
- Final validation accuracy: 99.46%
- Consistent improvement in both training and validation metrics
- Early stopping implemented at target accuracy of 99.4%

## Installation
pip install -r requirements.txt

## Training Results

The model achieves excellent performance on M1/M2 Mac using MPS acceleration:
Indiviudal model files contain the best results for that model and with Analyis and details

## Usage

To train the model:

```bash
python train.py
```
To run tests:

```bash
python test.py
```

## Model Architecture

- Model 1: 3 convolutional blocks with Linear layer
- Model 2: 3 convolutional blocks with BatchNorm and Linear layer
- Model 3: 3 convolutional blocks with BatchNorm and Dropout and Linear layer
- Model 4: 3 convolutional blocks with BatchNorm and Dropout and GAP 

## Augmentation
- Random Rotation

# Training 
- Adam optimizer
- Cyclic Learning Rate

## License

MIT License

## Training Logs
- With BatchNorm: 99.43%
- With Dropout: 99.52%
- With GAP: 99.46

## Running on Amazon EC2

### 1. Launch EC2 Instance
```bash
# Recommended: Deep Learning AMI with PyTorch
# Instance type: at least t2.medium (or g4dn.xlarge for GPU)
```

### 2. Connect to Instance
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

### 3. Setup Environment
```bash
# Clone repository
git clone https://github.com/your-repo/mnist_under_8k.git
cd mnist_under_8k

# Run setup script
chmod +x setup_ec2.sh
./setup_ec2.sh
```

### 4. Run Training
```bash
# Activate environment
source mnist_env/bin/activate

# Start training
python src/train.py

# For notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

### 5. Access Jupyter Notebook
- Note the token from jupyter output
- In your local browser: http://your-ec2-ip:8888
- Enter the token when prompted
