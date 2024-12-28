import torch
import torch.nn as nn
import torch.nn.functional as F

class NetReLU(nn.Module):
    """
    Basic CNN with only ReLU activations
    
    Target:
    - Establish baseline performance without regularization
    - Keep parameters under 8K
    - Understand impact of using only ReLU activations
    
    Analysis:
    - Simplest model in the series
    - No regularization techniques
    - Most prone to overfitting
    - Fastest training due to minimal complexity
    
    Results:
    - Training accuracy: ~99.2%
    - Test accuracy: ~98.6%
    - Parameters: 7,824
    - Prone to overfitting after epoch 10
    
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
        super(NetReLU, self).__init__()
        
        # First block: RF 1→3→5→6
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                     out_channels=8, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, 
                     out_channels=8, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
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
            nn.Conv2d(in_channels=16, 
                     out_channels=16, 
                     kernel_size=3, 
                     padding=1, 
                     stride=1),
            nn.ReLU(),
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
            nn.Conv2d(in_channels=16, 
                     out_channels=16, 
                     kernel_size=1),  # 1x1 conv
            nn.ReLU(),
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