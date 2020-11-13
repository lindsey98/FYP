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


index = 419
setup_seed(1914)
batch_size = 1000

model_before = CIFAR_17().cuda()
state_dict = torch.load('./CIFAR-17-1.pt')
model_before.load_state_dict(state_dict)
model_before.eval()
correct_before_sum = 0

model_after = CIFAR_17().cuda()
state_dict = torch.load('./CNN-123-50.pt')
model_after.load_state_dict(state_dict)
model_after.eval()
correct_after_sum = 0

correct_before_from_after_feature_sum = 0

train_data_loader = cifar10_data_loader_train(batch_size, shuffle=False)

regression_dict = {
    (True, True): 0,
    (True, False): 0,
    (False, False): 0,
    (False, True): 0,
}

for data, target in train_data_loader:
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

    output_before_from_after_feature, _ = model_before.dense(features_after[2])
    pred_before_from_after_feature = output_before_from_after_feature.argmax(dim=1, keepdim=True)
    correct_before_from_after_feature = pred_before_from_after_feature.eq(target.cuda().view_as(pred_before_from_after_feature))
    correct_before_from_after_feature_sum += correct_before_from_after_feature.sum().item()

    loss_value = list()
    for i in [2]:
        resize_feature = features_before[i].view(features_before[i].size(0), features_before[i].size(1), -1).mean(axis=2)
        resize_ref_feature = features_after[i].view(features_after[i].size(0), features_after[i].size(1), -1).mean(axis=2)

        normalize_feature_before = F.normalize(resize_feature, dim=1)
        normalize_feature_after = F.normalize(resize_ref_feature, dim=1)

        print(correct_before[index])
        print(correct_after[index])
        print(correct_before_from_after_feature[index])

        # print(features_before[i][index])
        # print(features_after[i][index])

        # print(torch.mul(features_before[i], features_after[i]).sum(dim=1)[index])
        print(resize_feature[index])
        print(resize_ref_feature[index])
        #
        print(normalize_feature_before[index])
        print(normalize_feature_after[index])
        temp_loss = torch.mul(normalize_feature_before, normalize_feature_after).sum(dim=1)
        print(temp_loss[index])
        #
        print(correct_before_sum)
        print(correct_after_sum)

        for i in range(correct_before.size(0)):
            # regression_dict[(correct_after[i].item(), correct_before_from_after_feature[i].item())] += 1
            if correct_before[i].item() == False and correct_after[i].item() == False and correct_before_from_after_feature[i] == True:
                print(1)
        # print(correct_before_from_after_feature_sum)

        # print((correct_before == correct_before_from_after_feature).int().sum())
        # print((correct_after == correct_before_from_after_feature).int().sum())
    break

print(regression_dict)