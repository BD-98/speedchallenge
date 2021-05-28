from torch.utils.data import Dataset
import numpy as np 
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.transforms import ToPILImage, ToTensor


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


if __name__ == "__main__":
    trainset = OFDataset(root="data/of-train-images-color", transform=ToTensor())
    loader = DataLoader(trainset, batch_size=3, shuffle=False)
    x,y = iter(loader).next()
    print(y)