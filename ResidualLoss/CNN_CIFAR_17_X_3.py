import random
import sys

from torch import optim

import numpy as np
from torch.backends import cudnn
import torch.nn.functional as F
import torch

from ResidualLoss.dataset import cifar10_data_loader_test, cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17_X


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
        pass


sys.stdout = Logger()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


setup_seed(1914)
num_epochs = 5000
batch_size = 100
learning_rate = 0.0001

train_data_loader = cifar10_data_loader_train(batch_size)
train_data_length = 50000
test_data_loader = cifar10_data_loader_test(batch_size)


def residual_train(l1, l2, l3):
    print(l1, l2, l3)
    total_correct_sum = 0
    total_classification_loss = 0

    model = CIFAR_17_X(l1, l2, l3).cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        total_correct = 0
        model.train()
        total_train_loss = 0

        for data, target in train_data_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()

            loss.backward()
            optimizer.step()

            total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        total_train_loss /= train_data_length
        total_correct_sum += total_correct
        total_classification_loss += total_train_loss
        print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, train_data_length))
        if (epoch + 1) % 1000 == 0:
            torch.save(model.state_dict(), "./CNN-Train/CIFAR_17_X/%s-%s-%s-epoch-%s.pt" % (l1, l2, l3, (epoch + 1)))
    print(l1, l2, l3)
    print("average correct:", total_correct_sum / num_epochs)
    print("average loss:", total_classification_loss / num_epochs)

# 1000, 500, 200, 100, 75, 50, 25, 10, 5, 1, 0.5,


if __name__ == '__main__':
    lst = [
        (9, 9, 8),
        (9, 9, 9),
        (9, 9, 10),
        (9, 10, 8),
    ]
    for a, b, c in lst:
        residual_train(a, b, c)

