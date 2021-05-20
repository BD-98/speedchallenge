import sys 
sys.path.append("GMA/core")

from network import RAFTGMA

from helpers import DEVICE, flow_to_image, load_image, InputPadder, viz
import torch 
import argparse

def demo(args):
    model: torch.nn.Module = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load("GMA/checkpoints/gma-things.pth"))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        test_img1 = load_image("data/train-frames/10.jpg")
        test_img2 = load_image("data/train-frames/11.jpg")
        padder = InputPadder(test_img1.shape)
        test_img1, test_img2 = padder.pad(test_img1, test_img2)
        flow_low, flow_up = model(test_img1, test_img2, iters=12, test_mode=True)
        viz(test_img1, flow_low)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_heads", type=int, default=1, help="number of heads in attention aggregator")
    parser.add_argument("--mixed_precision", action='store_true', help='use mixed precision')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    args = parser.parse_args()
    demo(args)