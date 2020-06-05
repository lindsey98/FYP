import os
import numpy as np
from PIL import Image, ImageOps
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from torch import nn, autograd
import torch.nn.functional as F


emb_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def l2_norm(x):
    if len(x.shape):
        x = x.reshape((x.shape[0], -1))
    return F.normalize(x, p=2, dim=1)


class MarginNet(nn.Module):
    def __init__(self, base_net, emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=(1, 1))
        self.base_net = base_net
        in_dim = base_net.fc.out_features
        self.emb_dim = emb_dim
        self.dense1 = nn.Linear(in_dim, emb_dim)
        self.normalize = l2_norm

    def forward(self, x):
        x = self.conv(x)
        x = self.base_net(x)  # renet18 embedding
        x = self.dense1(x)

        x = self.normalize(x)
        return x


def load_model():
    basenet = models.__dict__['resnet50'](pretrained=True)
    basenet.fc = nn.Linear(2048, 256)
    ct = 0
    for child in basenet.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    model = MarginNet(base_net=basenet, emb_dim=emb_dim)
    return model.to(device)


def procees_image(img_path):
    with torch.no_grad():
        img = Image.open(os.path.join(img_path)).convert("L")
        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2, (max(img.size) - img.size[0]) // 2,
            (max(img.size) - img.size[1]) // 2), fill=255)
        img = img.resize((100, 100))
        img = np.asarray(img)

        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        img = img.unsqueeze(0)
        return Variable(img.to(device), requires_grad=True)
