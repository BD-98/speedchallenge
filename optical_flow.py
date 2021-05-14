import sys 
sys.path.append("GMA/core")
import argparse
import torch 
import cv2
import matplotlib.pyplot as plt 
from network import RAFTGMA
from utils.utils import InputPadder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--num_heads", type=int, default=1, help="Number of heads for aggregate")
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')  
parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention') 
parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
args = parser.parse_args()

model = torch.nn.DataParallel(RAFTGMA(args))
model.load_state_dict(torch.load("GMA/checkpoints/gma-things.pth", map_location=device))
test_img1 = torch.tensor(cv2.imread("data/train-frames/10.jpg")).permute(2,1,0).unsqueeze(0)
test_img2 = torch.tensor(cv2.imread("data/train-frames/11.jpg")).permute(2,1,0).unsqueeze(0)
padder = InputPadder(test_img1.shape)
test_img1, test_img2 = padder.pad(test_img1, test_img2)
print(test_img1.shape)