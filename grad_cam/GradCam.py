import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

from siamese.Siamese import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model_name = '../siamese/dws_checkpoint_gray_v6.pth.tar'
    if device == 'cpu':
        state_dict = torch.load(model_name, map_location='cpu')
    else:
        state_dict = torch.load(model_name)

    pair_path = "../siamese/data/D_S/0/"

    img_s_path = pair_path + 'logo.png'
    img_o_path = pair_path + 'yolo_box.png'

    model = load_model()
    model.to(device)
    model.load_state_dict(state_dict)

    for name, module in model._modules.items():
        print(name)
        print(module)

    print(model._modules["base_net"])
