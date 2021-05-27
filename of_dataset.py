from torch.utils.data import Dataset, DataLoader
import numpy as np 
import cv2 
import torch 
import matplotlib.pyplot as plt 
a = np.load("data/train-of-frames/10.npy").squeeze(0)
print(a.shape)