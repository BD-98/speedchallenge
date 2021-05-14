import sys 
sys.path.append("GMA/core")
import argparse
import torch 
import cv2

from network import RAFTGMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--num_heads", type=int, default=1, help="Number of heads for aggregate")
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')  
parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention') 
parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
args = parser.parse_args()

model: torch.nn.Module = RAFTGMA(args).to(device)
test_img1 = torch.tensor(cv2.imread("data/train-frames/10.jpg")).permute(2,1,0).unsqueeze(0)
test_img2 = torch.tensor(cv2.imread("data/train-frames/11.jpg")).permute(2,1,0).unsqueeze(0)
out = model(test_img1, test_img2)
print(out[0].shape)