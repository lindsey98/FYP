import random

import numpy as np
from torch.backends import cudnn
import torch
import torch.nn.functional as F

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
eval_batch_size = 10000

model = CIFAR_17().cuda()
state_dict = torch.load('./CIFAR-17-1.pt')
model.load_state_dict(state_dict)
model.eval()
correct_sum = 0

eval_data_loader = cifar10_data_loader_train(eval_batch_size, shuffle=False)

grad_list = list()
correct_list = list()
count = 0
for data, target in eval_data_loader:
    data, target = data.cuda(), target.cuda()

    output, features = model.features(data)
    loss = F.nll_loss(output, target)
    pred = output.argmax(dim=1)
    correct = pred.eq(target)

    model.zero_grad()
    features[2].grad = None

    loss.sum().backward()
    grad_list.append(features[2].grad.view(features[2].size(0), features[2].size(1), -1).mean(axis=2).detach().clone().cpu().numpy())
    correct_list.append(correct.detach().clone().cpu().numpy())

grads = np.concatenate(grad_list, axis=0)
corrects = np.concatenate(correct_list, axis=0)

np.save('CD/8-dim', grads)
np.save('CD/correct', corrects)