import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImagenetNormWrapper(Dataset):
    """
    A wrapper dataset that applies ImageNet normalization to the 'rgb' field of the data.
    Assumes 'rgb' is a numpy array with shape (3, H, W) and values in [0, 255].
    Converts to [0, 1], applies normalization, and returns as PyTorch tensor.
    
    Uses standard ImageNet normalization values:
    - mean: [0.485, 0.456, 0.406]
    - std: [0.229, 0.224, 0.225]
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Standard ImageNet normalization values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        rgb = data['rgb']
        
        # Assuming rgb is numpy array (3, H, W), uint8 in [0, 255]
        # Convert to float32, scale to [0, 1], then to tensor
        rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb)
        
        # Apply normalization
        data['rgb'] = self.normalize(rgb)
        return data
