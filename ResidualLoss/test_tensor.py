import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import torch.nn.functional as F

import numpy as np
from NeuronCoverage.model import MINST_3, MINST_8
from NeuronCoverage.neuron_coverage_model import NeuronCoverageReLUModel
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = FashionMNIST('../data', transform=img_transform, train=False, download=True)

print( F.binary_cross_entropy(torch.Tensor([0.4]), torch.Tensor([0.6]), reduction='none'))
