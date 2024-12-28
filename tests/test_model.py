import unittest
import torch
import sys
import os
from torchsummary import summary
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_gap import NetGAP
from src.utils import count_parameters

# Add at the start of your test file
RUNNING_IN_CI = os.environ.get('CI') == 'true'

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = NetGAP()
        print("\nInitialized model for testing...")

    def test_torch_summary(self):
        print(f"Model summary: {summary(self.model, (1, 28, 28))}")
    
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        print(f"\nTotal parameters in model: {param_count:,}")
        print(f"Parameter limit: 8,000")
        self.assertLess(param_count, 8000, 
                       f"Model has {param_count:,} parameters, which exceeds the limit of 8,000")
    
    def test_batch_norm_exists(self):
        bn_layers = [module for module in self.model.modules() 
                    if isinstance(module, torch.nn.BatchNorm2d)]
        print(f"\nFound {len(bn_layers)} BatchNorm layers in model:")
        for idx, layer in enumerate(bn_layers, 1):
            print(f"BatchNorm layer {idx}: {layer}")
        
        has_batch_norm = len(bn_layers) > 0
        self.assertTrue(has_batch_norm, "Model should contain batch normalization layers")
    
    def test_dropout_exists(self):
        dropout_layers = [module for module in self.model.modules() 
                         if isinstance(module, torch.nn.Dropout)]
        print(f"\nFound {len(dropout_layers)} Dropout layers in model:")
        for idx, layer in enumerate(dropout_layers, 1):
            print(f"Dropout layer {idx}: {layer}")
        
        has_dropout = len(dropout_layers) > 0
        self.assertTrue(has_dropout, "Model should contain dropout layers")
    
    def test_gap_or_fc_exists(self):
        gap_layers = [module for module in self.model.modules() 
                     if isinstance(module, torch.nn.AdaptiveAvgPool2d)]
        fc_layers = [module for module in self.model.modules() 
                    if isinstance(module, torch.nn.Linear)]
        
        print("\nChecking for GAP or FC layers:")
        print(f"Found {len(gap_layers)} Global Average Pooling layers:")
        for idx, layer in enumerate(gap_layers, 1):
            print(f"GAP layer {idx}: {layer}")
            
        print(f"\nFound {len(fc_layers)} Fully Connected layers:")
        for idx, layer in enumerate(fc_layers, 1):
            print(f"FC layer {idx}: {layer}")
        
        has_gap = len(gap_layers) > 0
        has_fc = len(fc_layers) > 0
        self.assertTrue(has_gap or has_fc, 
                       "Model should contain either Global Average Pooling or Fully Connected layers")
    
    @unittest.skipIf(RUNNING_IN_CI, "Skipping GPU tests in CI environment")
    def test_gpu_specific_functionality(self):
        # GPU tests here
        pass

if __name__ == '__main__':
    unittest.main() 