from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_data_loaders(batch_size=64):

    train_transform=transforms.Compose([
                        transforms.Resize((28, 28)),
                        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create indices for 25% of training data
    # total_train = len(train_dataset)
    # indices = np.random.permutation(total_train)
    # train_size = int(0.25 * total_train)  # 25% of the data
    # train_indices = indices[:train_size]
    
    # Create subset of training data
    # train_dataset = Subset(train_dataset, train_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # print(f"Training with {train_size:,} samples (25% of original {total_train:,} samples)")
    print(f"Training with {len(train_dataset)} samples")

    
    return train_loader, test_loader 