import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F

from TBNN.model import MINST_3, MINST_8
from TBNN.neuron_coverage_model import NeuronCoverageReLUModel

a = torch.load("MINST-3.pth")
print(a)

b = torch.load("MINST-3-A.pth")
print(b)
