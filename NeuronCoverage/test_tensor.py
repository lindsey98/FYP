import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F

import numpy as np
from NeuronCoverage.model import MINST_3, MINST_8
from NeuronCoverage.neuron_coverage_model import NeuronCoverageReLUModel
from ResidualLoss.dataset import image_net_data_loader_train

# train_data_loader = image_net_data_loader_train(128)
#
# for data in train_data_loader:
#     img, target = data
#     img = Variable(img.view(img.size(0), -1)).cuda()
#     print(img.shape)
#     break

print(np.stack([np.array([1, 2]), 1]))
