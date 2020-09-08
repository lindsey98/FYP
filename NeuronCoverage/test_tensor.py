import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F

import numpy as np
from NeuronCoverage.model import MINST_3, MINST_8
from NeuronCoverage.neuron_coverage_model import NeuronCoverageReLUModel

a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([5, 6])

torch.save((a, b), "./t.pt")
t = torch.load("./t.pt")

print(t[0])
print(t[1])
# print(torch.load("../data/FashionMNIST/processed/test.pt"))