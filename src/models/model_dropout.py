import torch
import torch.nn as nn
import torch.nn.functional as F

DROP_OUT = 0.05

class NetDropout(nn.Module):
    """
    CNN with BatchNorm and Dropout

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
    
    Target:
    - Add dropout regularization to BatchNorm model
    - Prevent co-adaptation of features
    - Improve generalization over BatchNorm model
    - Keep parameters ~ 8K
    
    Analysis:
    - Dropout rate 0.05 (found optimal through experimentation)
    - Dropout after ReLU and BatchNorm
    - Same parameter count as BatchNorm model
    - Multiple Drop out layers were not effective. so kept only one dropout layer per block
    - Best results with feature Augmentation, Cyclic LR and Adam
    
    Results:
    - Training accuracy: ~99.49%
    - Test accuracy: ~99.52%
    - Parameters: 8354
    - Better generalization than BatchNorm

----------------------------------------------------------------

Training With Dropout model...
Training with 60000 samples
Epoch: 0 | Train Loss: 0.425 | Train Acc: 87.66% | Val Loss: 6.092 | Val Acc: 97.07% | Best Val Acc: 97.07%
Epoch: 1 | Train Loss: 0.096 | Train Acc: 96.99% | Val Loss: 3.063 | Val Acc: 98.62% | Best Val Acc: 98.62%
Epoch: 2 | Train Loss: 0.081 | Train Acc: 97.42% | Val Loss: 3.079 | Val Acc: 98.53% | Best Val Acc: 98.62%
Epoch: 3 | Train Loss: 0.068 | Train Acc: 97.84% | Val Loss: 2.438 | Val Acc: 98.75% | Best Val Acc: 98.75%
Epoch: 4 | Train Loss: 0.058 | Train Acc: 98.14% | Val Loss: 2.310 | Val Acc: 98.78% | Best Val Acc: 98.78%
Epoch: 5 | Train Loss: 0.052 | Train Acc: 98.34% | Val Loss: 2.028 | Val Acc: 98.90% | Best Val Acc: 98.90%
Epoch: 6 | Train Loss: 0.049 | Train Acc: 98.44% | Val Loss: 2.755 | Val Acc: 98.60% | Best Val Acc: 98.90%
Epoch: 7 | Train Loss: 0.042 | Train Acc: 98.69% | Val Loss: 1.508 | Val Acc: 99.14% | Best Val Acc: 99.14%
Epoch: 8 | Train Loss: 0.039 | Train Acc: 98.78% | Val Loss: 1.266 | Val Acc: 99.28% | Best Val Acc: 99.28%
Epoch: 9 | Train Loss: 0.032 | Train Acc: 98.96% | Val Loss: 1.382 | Val Acc: 99.28% | Best Val Acc: 99.28%
Epoch: 10 | Train Loss: 0.029 | Train Acc: 99.08% | Val Loss: 1.130 | Val Acc: 99.37% | Best Val Acc: 99.37%

Reached target accuracy of 99.4% at epoch 11
Epoch: 11 | Train Loss: 0.023 | Train Acc: 99.25% | Val Loss: 1.090 | Val Acc: 99.46% | Best Val Acc: 99.46%

Reached target accuracy of 99.4% at epoch 12
Epoch: 12 | Train Loss: 0.020 | Train Acc: 99.32% | Val Loss: 0.944 | Val Acc: 99.52% | Best Val Acc: 99.52%

Reached target accuracy of 99.4% at epoch 13
Epoch: 13 | Train Loss: 0.017 | Train Acc: 99.45% | Val Loss: 0.909 | Val Acc: 99.51% | Best Val Acc: 99.52%

Reached target accuracy of 99.4% at epoch 14
Epoch: 14 | Train Loss: 0.016 | Train Acc: 99.49% | Val Loss: 0.933 | Val Acc: 99.51% | Best Val Acc: 99.52%    

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
    """
    def __init__(self):
        super(NetDropout, self).__init__()
        
        # First block: RF 1→3→5→6
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                     out_channels=8, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout(p=DROP_OUT),
            nn.Conv2d(in_channels=8, 
                     out_channels=8, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout(p=DROP_OUT),
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
            nn.Dropout(p=DROP_OUT),
            nn.Conv2d(in_channels=16, 
                     out_channels=16, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout(p=DROP_OUT),
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
            nn.Dropout(p=DROP_OUT),
            nn.Conv2d(in_channels=16, 
                     out_channels=16, 
                     kernel_size=1),  # 1x1 conv
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout(p=DROP_OUT),
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
