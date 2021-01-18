import random
from torch.autograd import Variable
from torch import optim

import numpy as np
from torch.backends import cudnn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ResidualLoss.dataset import cifar10_data_loader_test, cifar10_data_loader_train, cifar10_dataset_train
from ResidualLoss.model import CIFAR_17


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


setup_seed(1914)
num_epochs = 200
batch_size = 100
evaluation_batch_size = 2000
learning_rate = 0.0001
alpha = 0.05
priority = 2

ref_model = CIFAR_17().cuda()
model = CIFAR_17().cuda()
state_dict = torch.load('./CIFAR-17-1.pt')
ref_model.eval()
model.train()

optimizer = optim.Adam([
    {'params': model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters()}
], lr=learning_rate, weight_decay=1e-5)

train_dataset = cifar10_dataset_train()
train_data_length = len(train_dataset)
sampler = WeightedRandomSampler([1] * train_data_length, num_samples=train_data_length, replacement=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False)
test_data_loader = cifar10_data_loader_test(batch_size)


def residual_train():
    prob = torch.ones(len(train_dataset), dtype=torch.float64)
    start_index = 0
    for data, target in evaluation_data_loader:
        data, target = data.cuda(), target.cuda()
        output = ref_model(data)
        pred = output.argmax(dim=1)
        correct_list = pred.eq(target)

        for i in range(correct_list.size(0)):
            if not correct_list[i].item():
                prob[start_index + i] = priority

        start_index += evaluation_batch_size

    sampler.weights = prob
    print(prob.sum())

    total_correct_sum = 0
    total_classification_loss = 0

    for epoch in range(num_epochs):
        total_correct = 0
        model.eval()

        with torch.no_grad():
            for data, target in evaluation_data_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total_correct += pred.eq(target.view_as(pred)).sum().item()

        model.train()
        total_train_loss = 0
        for data, target in train_data_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output, features = model.features(data)

            ref_output, ref_features = ref_model.features(data)
            ref_pred = ref_output.argmax(dim=1)
            ref_list = 2 * ref_pred.eq(target).int() - 1

            loss1 = 0
            for i in [2]:
                resize_feature = features[i].view(features[i].size(0), features[i].size(1), -1).mean(axis=2)
                resize_ref_feature = ref_features[i].view(ref_features[i].size(0), ref_features[i].size(1), -1).mean(axis=2)

                normalize_ref_feature = F.normalize(resize_ref_feature, dim=1).detach()
                normalize_feature = F.normalize(resize_feature, dim=1)

                temp_loss = torch.norm(normalize_ref_feature - normalize_feature, p=2, dim=1)
                loss1 += (temp_loss * ref_list).sum()

            loss = F.nll_loss(output, target)
            loss += alpha * loss1

            loss.backward()
            optimizer.step()

            total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        total_train_loss /= train_data_length
        total_correct_sum += total_correct
        total_classification_loss += total_train_loss
        if epoch % 100 == 0:
            print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, train_data_length))
            test()

    print("average correct:", total_correct_sum / num_epochs)
    print("average loss:", total_classification_loss / num_epochs)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_data_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))

# 1000, 500, 200, 100, 75, 50, 25, 10, 5, 1, 0.5,


if __name__ == '__main__':
    for iter_1 in [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01]:
        for iter_2 in [1, 1.5, 2, 5, 10, 50, 100]:
            alpha = iter_1
            priority = iter_2
            print(alpha, priority)
            ref_model.load_state_dict(state_dict)
            model.load_state_dict(state_dict)
            residual_train()
            loc = "./CNN-l2-freeze-upperbound/alpha-%s-p-%s.pt" % (alpha, priority)
            torch.save(model.state_dict(), loc)
            print(alpha, priority)
