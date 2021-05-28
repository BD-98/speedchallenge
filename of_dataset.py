from torch.utils.data import Dataset
import numpy as np 
import torch
from torchvision.datasets import ImageFolder


class OFDataset(Dataset):
    def __init__(self, root, transform, train=True):
        self.data = ImageFolder(root, transform=transform)
        self.train = train
    
    def __getitem__(self, idx):
        if self.train:
            images, _ = self.data[idx]
            labels = torch.tensor(np.loadtxt("data/train.txt"))[idx]
            return (images, labels)
        else:
            return self.data[idx]    
    def __len__(self):
        return len(self.data)


