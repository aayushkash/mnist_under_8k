import torch
import torch.nn as nn
import torch.nn.functional as F

class NetBatchNorm(nn.Module):
    """
    CNN with BatchNorm after each convolution
    
    Target:
    - Improve training stability over base ReLU model
    - Reduce internal covariate shift
    - Keep parameters close to 8K while adding BatchNorm
    - Achieve faster convergence than ReLU model
    ==================================================
Model Architecture: With Dropout
==================================================
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
            Conv2d-4            [-1, 8, 28, 28]             584
              ReLU-5            [-1, 8, 28, 28]               0
       BatchNorm2d-6            [-1, 8, 28, 28]              16
         MaxPool2d-7            [-1, 8, 14, 14]               0
           Dropout-8            [-1, 8, 14, 14]               0
            Conv2d-9           [-1, 16, 14, 14]           1,168
             ReLU-10           [-1, 16, 14, 14]               0
      BatchNorm2d-11           [-1, 16, 14, 14]              32
           Conv2d-12           [-1, 16, 14, 14]           2,320
      BatchNorm2d-13           [-1, 16, 14, 14]              32
             ReLU-14           [-1, 16, 14, 14]               0
        MaxPool2d-15             [-1, 16, 7, 7]               0
          Dropout-16             [-1, 16, 7, 7]               0
           Conv2d-17             [-1, 16, 7, 7]           2,320
             ReLU-18             [-1, 16, 7, 7]               0
      BatchNorm2d-19             [-1, 16, 7, 7]              32
          Dropout-20             [-1, 16, 7, 7]               0
           Conv2d-21             [-1, 16, 7, 7]             272
             ReLU-22             [-1, 16, 7, 7]               0
      BatchNorm2d-23             [-1, 16, 7, 7]              32
        MaxPool2d-24             [-1, 16, 3, 3]               0
          Dropout-25             [-1, 16, 3, 3]               0
           Linear-26                   [-1, 10]           1,450
================================================================
Total params: 8,354
----------------------------------------------------------------
    
    Analysis:
    - Adds BatchNorm after each convolution
    - Slightly more parameters than ReLU model
    - Better training stability
    - Order: Conv -> ReLU -> BatchNorm for optimal performance
    - Better results with ADAM and Cyclic Learning Rate 
    
    Layer details:
--------------------------------------------------
    Results:
    - Training accuracy: ~99.76%
    - Test accuracy: ~99.43%
    - Parameters: 8,354
    - Faster convergence than ReLU model
    - Better generalization than base model
--------------------------------------------------
Training Logs:
Training With BatchNorm model...
Training with 60000 samples
Epoch: 0 | Train Loss: 0.351 | Train Acc: 90.02% | Val Loss: 5.356 | Val Acc: 97.34% | Best Val Acc: 97.34%
Epoch: 1 | Train Loss: 0.076 | Train Acc: 97.56% | Val Loss: 4.255 | Val Acc: 97.84% | Best Val Acc: 97.84%
Epoch: 2 | Train Loss: 0.072 | Train Acc: 97.78% | Val Loss: 2.536 | Val Acc: 98.66% | Best Val Acc: 98.66%
Epoch: 10 | Train Loss: 0.020 | Train Acc: 99.36% | Val Loss: 1.475 | Val Acc: 99.24% | Best Val Acc: 99.24%
Epoch: 11 | Train Loss: 0.014 | Train Acc: 99.53% | Val Loss: 1.318 | Val Acc: 99.36% | Best Val Acc: 99.36%
Epoch: 12 | Train Loss: 0.011 | Train Acc: 99.61% | Val Loss: 1.165 | Val Acc: 99.38% | Best Val Acc: 99.38%

Reached target accuracy of 99.4% at epoch 13
Epoch: 13 | Train Loss: 0.008 | Train Acc: 99.75% | Val Loss: 1.138 | Val Acc: 99.43% | Best Val Acc: 99.43%

Reached target accuracy of 99.4% at epoch 14
Epoch: 14 | Train Loss: 0.008 | Train Acc: 99.76% | Val Loss: 1.135 | Val Acc: 99.41% | Best Val Acc: 99.43%

--------------------------------------------------
Receptive Field (RF) calculation:
    RF = 1 + sum((kernel_size - 1) * stride_product)
    stride_product = product of all previous strides
    
    Layer details:
    Conv1: RF_in=1, k=3, s=1, p=1 → RF_out=3
    Conv2: RF_in=3, k=3, s=1, p=1 → RF_out=5
    MaxPool1: RF_in=5, k=2, s=2 → RF_out=6
    
    Conv3: RF_in=6, k=3, s=1, p=1 → RF_out=10
    Conv4: RF_in=10, k=3, s=1, p=1 → RF_out=14
    MaxPool2: RF_in=14, k=2, s=2 → RF_out=16
    
    Conv5: RF_in=16, k=3, s=1, p=1 → RF_out=20
    Conv6: RF_in=20, k=1, s=1 → RF_out=20
    MaxPool3: RF_in=20, k=2, s=2 → RF_out=22
    
    Final RF = 22x22 pixels
--------------------------------------------------
    """
    def __init__(self):
        super(NetBatchNorm, self).__init__()
        
        # First block: RF 1→3→5→6
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                     out_channels=8, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, 
                     out_channels=8, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2
        )

        # Second block: RF 6→10→14→16
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, 
                     out_channels=16, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, 
                     out_channels=16, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2
        )

        # Third block: RF 16→20→20→22
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                     out_channels=16, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, 
                     out_channels=16, 
                     kernel_size=1),  # 1x1 conv
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * 3 * 3, 
                     out_features=10)
        )
        
    def forward(self, x):
        x = self.conv1(x)  # 28x28 -> 14x14
        x = self.conv2(x)  # 14x14 -> 7x7
        x = self.conv3(x)  # 7x7 -> 3x3
        x = x.view(-1, 16 * 3 * 3)
        x = self.fc(x)     # 144 -> 10
        return F.log_softmax(x, dim=1) 