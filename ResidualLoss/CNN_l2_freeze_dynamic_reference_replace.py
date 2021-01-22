import random
import sys

from torch.autograd import Variable
from torch import optim

import numpy as np
from torch.backends import cudnn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from ResidualLoss.dataset import cifar10_data_loader_test, cifar10_data_loader_train, L2Dataset, cifar10_dataset_train
from ResidualLoss.model import CIFAR_17


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        log_loc = "./log/%s.txt" % sys.argv[0].split("/")[-1].split(".")[0]
        self.log = open(log_loc, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()


sys.stdout = Logger()


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
evaluation_batch_size = 100
learning_rate = 0.0001
alpha = 0.05

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


train_dataset = L2Dataset(cifar10_dataset_train())
train_data_length = len(train_dataset)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size)

test_data_loader = cifar10_data_loader_test(batch_size, shuffle=False)
evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False)


def residual_train():
    start_index = 0
    for data, target in evaluation_data_loader:
        data, target = data.cuda(), target.cuda()
        output, features = ref_model.features(data)
        pred_before = output.argmax(dim=1)
        correct = pred_before.eq(target)

        normalize_ref_feature = F.normalize(features[2].view(features[2].size(0), features[2].size(1), -1).mean(axis=2), dim=1).detach().cpu()

        nll_loss = F.nll_loss(output, target, reduction='none')

        for i in range(correct.size(0)):
            if correct[i].item() and nll_loss[i] < train_dataset.l2_loss[start_index + i]:
                train_dataset.l2_ref[start_index + i] = normalize_ref_feature[i]

        start_index += evaluation_batch_size

    total_correct_sum = 0
    total_classification_loss = 0
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_correct = 0
        for data, target, normalize_ref_feature, idx in train_data_loader:
            data, target, normalize_ref_feature = data.cuda(), target.cuda(), normalize_ref_feature.cuda()

            optimizer.zero_grad()
            output, features = model.features(data)

            loss1 = 0

            resize_feature = features[2].view(features[2].size(0), features[2].size(1), -1).mean(axis=2)
            normalize_feature = F.normalize(resize_feature, dim=1)

            valid_list = (normalize_ref_feature.sum(dim=1) != 0).int()
            loss1 += (torch.norm(normalize_ref_feature - normalize_feature, p=2, dim=1) * valid_list).sum()

            loss = F.nll_loss(output, target, reduction='none')
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct = pred.eq(target)

            for i in range(data.size(0)):
                if correct[i].item() and loss[i] < train_dataset.l2_loss[idx[i]]:
                    train_dataset.l2_ref[idx[i]] = normalize_feature[i].detach().cpu()

            final_loss = loss.sum() + alpha * loss1

            final_loss.backward()
            optimizer.step()
            total_train_loss += loss.sum().item()  # sum up batch loss
            total_correct += correct.sum().item()

        total_train_loss /= train_data_length
        total_correct_sum += total_correct
        total_classification_loss += total_train_loss
        if epoch % 50 == 0:
            print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, train_data_length))
            test()
        ref_model.load_state_dict(model.state_dict())

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
    for j in [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]:
        alpha = j
        print(alpha)
        train_dataset.reset()
        ref_model.load_state_dict(state_dict)
        model.load_state_dict(state_dict)
        residual_train()
        loc = "./CNN-l2-dynamic-reference/replace-alpha-" + str(j) + ".pt"
        torch.save(model.state_dict(), loc)
        print((train_dataset.l2_loss != 0).int().sum())
        print(alpha)
