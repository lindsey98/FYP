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
batch_size = 10000

model_before = CIFAR_17().cuda()
state_dict = torch.load('./CIFAR-17-1.pt')
model_before.load_state_dict(state_dict)
model_before.eval()
correct_before_sum = 0

model_after = CIFAR_17().cuda()
state_dict = torch.load('./CNN-l2-upperbound/not-freeze-re-init.pt')
model_after.load_state_dict(state_dict)
model_after.eval()
correct_after_sum = 0

regression_dict = {
    (True, True): 0,
    (True, False): 0,
    (False, False): 0,
    (False, True): 0,
}

train_data_loader = cifar10_data_loader_train(batch_size)

for data, target in train_data_loader:
    data, target = data.cuda(), target.cuda()

    output_before = model_before(data)
    pred_before = output_before.argmax(dim=1)
    correct_before = pred_before.eq(target)
    correct_before_sum += correct_before.sum().item()

    output_after = model_after(data)
    pred_after = output_after.argmax(dim=1)
    correct_after = pred_after.eq(target)
    correct_after_sum += correct_after.sum().item()

    for i in range(correct_before.size(0)):
        regression_dict[(correct_before[i].item(), correct_after[i].item())] += 1

print(regression_dict)
print(correct_before_sum)
print(correct_after_sum)
