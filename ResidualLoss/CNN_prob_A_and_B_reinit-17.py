import random
import sys

from torch.autograd import Variable
from torch import optim

import numpy as np
from torch.backends import cudnn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import cifar10_data_loader_test, cifar10_data_loader_train, cifar10_dataset_train
from model import CIFAR_17


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
evaluation_batch_size = 100
learning_rate = 0.0001

model = CIFAR_17().cuda()
model.train()

# optimizer = optim.Adam([
#     {'params': model.conv1.parameters()},
#     {'params': model.conv2.parameters()},
#     {'params': model.conv3.parameters()}
# ], lr=learning_rate, weight_decay=1e-5)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

train_dataset = cifar10_dataset_train()
train_data_length = len(train_dataset)
sampler = WeightedRandomSampler([1] * train_data_length, num_samples=train_data_length, replacement=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False)
test_data_loader = cifar10_data_loader_test(batch_size)


def residual_train():
    prob = torch.zeros(len(train_dataset), dtype=torch.float64).cuda()
    _, init_train_list, init_not_train_list, after_train_list, after_not_train_list = torch.load(
        "./analysis-and-draw/data/CNN-30-CIFAR_17-A_B.pt")

    B_17 = set(init_train_list) - set(after_train_list)
    A_17 = set(after_not_train_list) - set(init_not_train_list)

    for idx in B_17:
        prob[idx] = 1
    for idx in A_17:
        prob[idx] = 1
    sampler.weights = prob
    print(prob.sum())

    total_correct_sum = 0
    total_classification_loss = 0

    for epoch in range(num_epochs):
        total_correct = 0
        total_correct_1 = 0
        model.eval()

        with torch.no_grad():
            start_index = 0
            for data, target in evaluation_data_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_lst = pred.eq(target.view_as(pred)).squeeze()
                total_correct += correct_lst.sum().item()
                
                for i in range(evaluation_batch_size):
                    if start_index + i in A_17 or start_index + i in B_17:
                        if correct_lst[i]:
                            total_correct_1 += 1
                start_index += evaluation_batch_size

        model.train()
        total_train_loss = 0
        for data, target in train_data_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output, features = model.features(data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        total_train_loss /= train_data_length
        total_correct_sum += total_correct
        total_classification_loss += total_train_loss
        print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{} Extra: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, train_data_length, total_correct_1, len(A_17) + len(B_17)))

    print("average correct:", total_correct_sum / num_epochs)
    print("average loss:", total_classification_loss / num_epochs)


if __name__ == '__main__':
    residual_train()
    loc = "./CNN-30/A_and_B-reinit-17.pt"
    torch.save(model.state_dict(), loc)
