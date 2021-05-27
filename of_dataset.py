from torch.serialization import load
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import cv2 
import torch 
import os 
from natsort import natsorted
from torchvision.transforms import transforms


class OFDataset(Dataset):
    def __init__(self, train=True):
        self.train_path = "data/of-train-images-color"
        self.test_path = "data/of-test-images-color"
        self.train = train
        
    
    def load_images(self):
        if self.train:
            train_images = natsorted([cv2.imread(str(os.path.join(self.train_path, img)) for img in os.listdir(self.train_path))])
            return torch.cat(train_images)
          
        else:
            test_images = natsorted(cv2.imread([str(os.path.join(self.test_path, img)) for img in os.listdir(self.test_path)]))
            return torch.cat(test_images)
        
    def __getitem__(self, index):
        data_images = self.load_images()[index]
        labels = torch.tensor(np.loadtxt("data/train.txt"))[index]
        if self.train:
            return (data_images, labels)
        else:
            return data_images
    
    def __len__(self):
        return len(self.load_images())



a = OFDataset()
print(a[0])



    
        