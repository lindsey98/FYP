import random

import numpy as np
from torch.backends import cudnn
import torch
import torch.nn.functional as F
from torch import optim

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_16


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


setup_seed(1914)
batch_size = 100

model_before = CIFAR_16().cuda()
state_dict = torch.load('./CIFAR-16-5723.pt')
model_before.load_state_dict(state_dict)
model_before.eval()
correct_before_sum = 0

model_after = CIFAR_16().cuda()
state_dict = torch.load('./CIFAR-16-loss-1.pt')
model_after.load_state_dict(state_dict)
model_after.train()
correct_after_sum = 0
optimizer = optim.Adam(model_after.parameters(), lr=0.0001, weight_decay=1e-5)

train_data_loader = cifar10_data_loader_train(batch_size)

for data, target in train_data_loader:
    optimizer.zero_grad()

    data, target = data.view(data.size(0), -1).cuda(), target.cuda()

    output_before, features_before = model_before.features(data)
    pred_before = output_before.argmax(dim=1, keepdim=True)
    correct_before = pred_before.eq(target.cuda().view_as(pred_before))
    correct_before_sum += correct_before.sum().item()

    ref_list = - 2 * correct_before.int() + 1

    output_after, features_after = model_after.features(data)
    pred_after = output_after.argmax(dim=1, keepdim=True)
    correct_after = pred_after.eq(target.cuda().view_as(pred_after))
    correct_after_sum += correct_after.sum().item()

    loss_value = list()
    for i in [0, 1, 2]:
        zeros = torch.zeros_like(features_before[i])
        dropped_ref_feature = torch.where(features_after[i] != 0, features_before[i], zeros)

        normalize_feature_before = F.normalize(dropped_ref_feature, dim=1)
        normalize_feature_after = F.normalize(features_after[i], dim=1)

        loss_value.append(torch.mul(normalize_feature_before, normalize_feature_after).sum(dim=1))

    # loss = F.nll_loss(output_after, target.cuda())

    loss = (loss_value[1] * ref_list).sum()
    loss.backward()
    print(model_after.hidden2[0].weight.grad)
    break



