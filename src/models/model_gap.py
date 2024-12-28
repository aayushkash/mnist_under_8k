import torch
import torch.nn as nn
import torch.nn.functional as F

DROP_OUT = 0.05

class NetGAP(nn.Module):
    """
    CNN with Global Average Pooling (GAP)
==================================================
Model Architecture: With GAP
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
             ReLU-13           [-1, 16, 14, 14]               0
      BatchNorm2d-14           [-1, 16, 14, 14]              32
        MaxPool2d-15             [-1, 16, 7, 7]               0
          Dropout-16             [-1, 16, 7, 7]               0
           Conv2d-17             [-1, 16, 7, 7]           2,320
             ReLU-18             [-1, 16, 7, 7]               0
      BatchNorm2d-19             [-1, 16, 7, 7]              32
           Conv2d-20             [-1, 10, 7, 7]             170
             ReLU-21             [-1, 10, 7, 7]               0
      BatchNorm2d-22             [-1, 10, 7, 7]              20
        MaxPool2d-23             [-1, 10, 3, 3]               0
          Dropout-24             [-1, 10, 3, 3]               0
AdaptiveAvgPool2d-25             [-1, 10, 1, 1]               0
================================================================
Total params: 6,790
================================================================


Target:
    - Replace FC layer with Global Average Pooling
    - Reduce parameter count significantly under 8k 
    - Maintain or improve accuracy compared to FC models
    - Make model Training With GAP layer with last epochs stable over 99.4%

Analysis:
    - GAP is a good replacement for FC layer with some compromises on results.
    - Initially we see a drop but later using Adam and Cyclic LR which improved the accuracy
    - Added Augmentation rotation from session notes to the data loader which improved the accuracy
    - I was able to reduced ~1400 parameters from previous dropout model
    - Slight tuning of drop out rate was done on cuda environments from my laptop Mac M2
    - 0.1 drop out rate was found to be optimal for this model on cuda

Results:  
  - I was able to achieve 99.46% accuracy on validation set on collab
    - I was able to achieve 99.45% accuracy on validation set on EC2

Training with 60000 samples
Epoch: 0 | Train Loss: 0.682 | Train Acc: 83.76% | Val Loss: 9.662 | Val Acc: 96.56% | Best Val Acc: 96.56%
Epoch: 1 | Train Loss: 0.177 | Train Acc: 95.61% | Val Loss: 5.702 | Val Acc: 97.65% | Best Val Acc: 97.65%
Epoch: 2 | Train Loss: 0.118 | Train Acc: 96.72% | Val Loss: 6.061 | Val Acc: 96.98% | Best Val Acc: 97.65%
Epoch: 3 | Train Loss: 0.100 | Train Acc: 97.06% | Val Loss: 3.488 | Val Acc: 98.21% | Best Val Acc: 98.21%
Epoch: 4 | Train Loss: 0.085 | Train Acc: 97.49% | Val Loss: 3.291 | Val Acc: 98.39% | Best Val Acc: 98.39%
Epoch: 5 | Train Loss: 0.074 | Train Acc: 97.83% | Val Loss: 2.967 | Val Acc: 98.55% | Best Val Acc: 98.55%
Epoch: 6 | Train Loss: 0.069 | Train Acc: 97.95% | Val Loss: 2.686 | Val Acc: 98.76% | Best Val Acc: 98.76%
Epoch: 7 | Train Loss: 0.060 | Train Acc: 98.22% | Val Loss: 2.716 | Val Acc: 98.62% | Best Val Acc: 98.76%
Epoch: 8 | Train Loss: 0.057 | Train Acc: 98.24% | Val Loss: 2.093 | Val Acc: 99.01% | Best Val Acc: 99.01%
Epoch: 9 | Train Loss: 0.051 | Train Acc: 98.43% | Val Loss: 1.725 | Val Acc: 99.19% | Best Val Acc: 99.19%
Epoch: 10 | Train Loss: 0.045 | Train Acc: 98.59% | Val Loss: 1.782 | Val Acc: 99.20% | Best Val Acc: 99.20%
Epoch: 11 | Train Loss: 0.039 | Train Acc: 98.81% | Val Loss: 1.606 | Val Acc: 99.15% | Best Val Acc: 99.20%
Epoch: 12 | Train Loss: 0.035 | Train Acc: 98.89% | Val Loss: 1.520 | Val Acc: 99.28% | Best Val Acc: 99.28%
Epoch: 13 | Train Loss: 0.032 | Train Acc: 99.03% | Val Loss: 1.446 | Val Acc: 99.29% | Best Val Acc: 99.29%
Epoch: 14 | Train Loss: 0.030 | Train Acc: 99.09% | Val Loss: 1.425 | Val Acc: 99.34% | Best Val Acc: 99.34% 
  
===================================================    
  Receptive Field (RF) calculation:
===================================================    

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

===================================================    

    """
    def __init__(self, dropout_rate=DROP_OUT):
        super(NetGAP, self).__init__()
        
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
            nn.Dropout(p=dropout_rate),
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
            nn.Dropout(p=dropout_rate),
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
                     out_channels=10, 
                     kernel_size=1,  # 1x1 conv for channel reduction
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2
            nn.Dropout(p=dropout_rate),
        )

        # Global Average Pooling: maintains RF while reducing spatial dims to 1x1
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        
    def forward(self, x):
        x = self.conv1(x)  # 28x28 -> 14x14
        x = self.conv2(x)  # 14x14 -> 7x7
        x = self.conv3(x)  # 7x7 -> 3x3
        x = self.gap(x)    # 3x3 -> 1x1
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)