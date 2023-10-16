import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from model import VAE


def predict(args):

    model = VAE()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.generate_image(args.out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--weights", type = str, default = 'weights/last.pt')
    parser.add_argument("--out", type = str, default = 'test/output_image.jpg')
    args = parser.parse_args()

    predict(args)