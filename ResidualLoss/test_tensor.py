import random

import numpy as np
from torch.backends import cudnn
import torch

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


setup_seed(1914)
batch_size = 100

model_before = CIFAR_17().cuda()
state_dict = torch.load('./CIFAR-17-1.pt')
model_before.load_state_dict(state_dict)
model_before.eval()
correct_before_sum = 0

train_data_loader = cifar10_data_loader_train(batch_size)
for data, target in train_data_loader:
    data, target = data.view(data.size(0), -1).cuda(), target.cuda()

    output_before, features = model_before.features(data)

    for i in features:
        print(i.shape)

    break

