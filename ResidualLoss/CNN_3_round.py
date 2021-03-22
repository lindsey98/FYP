import random
import sys

from torch.autograd import Variable
from torch import optim

import numpy as np
from torch.backends import cudnn
import torch.nn.functional as F
import torch

from ResidualLoss.dataset import cifar10_data_loader_test, cifar10_data_loader_train
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
num_epochs = 10
batch_size = 100
learning_rate = 0.0001
alpha = 0.05

ref_model = CIFAR_17().cuda()
model = CIFAR_17().cuda()
state_dict = torch.load('./CIFAR-17-1.pt')
ref_model.eval()
model.train()

optimizer_residual = optim.Adam([
    {'params': model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters()}
], lr=learning_rate, weight_decay=1e-5)

optimizer_dense = optim.Adam([
    {'params': model.dense1.parameters()},
    {'params': model.dense2.parameters()},
], lr=learning_rate, weight_decay=1e-5)

optimizer_whole = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

train_data_loader = cifar10_data_loader_train(batch_size)
test_data_loader = cifar10_data_loader_test(batch_size)


def residual_train():
    total_correct_sum_1 = 0
    total_correct_sum_2 = 0
    total_correct_sum_3 = 0
    total_classification_loss_1 = 0
    total_classification_loss_2 = 0
    total_classification_loss_3 = 0
    length = len(train_data_loader.dataset)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss_1 = 0
        total_correct_1 = 0

        for data in train_data_loader:
            img, target = data
            img = Variable(img.view(img.size(0), -1)).cuda()

            optimizer_residual.zero_grad()

            output, features = model.features(img)

            ref_output, ref_features = ref_model.features(img)
            ref_pred = ref_output.argmax(dim=1)
            ref_list = 2 * ref_pred.eq(target.cuda()).int() - 1

            loss1 = 0
            for i in [2]:
                resize_feature = features[i].view(features[i].size(0), features[i].size(1), -1).mean(axis=2)
                resize_ref_feature = ref_features[i].view(ref_features[i].size(0), ref_features[i].size(1), -1).mean(axis=2)

                normalize_ref_feature = F.normalize(resize_ref_feature, dim=1).detach()
                normalize_feature = F.normalize(resize_feature, dim=1)

                temp_loss = torch.norm(normalize_ref_feature - normalize_feature, p=2, dim=1)
                loss1 += (temp_loss * ref_list).sum()

            loss = F.nll_loss(output, target.cuda())
            loss += alpha * loss1

            loss.backward()
            optimizer_residual.step()

            total_train_loss_1 += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total_correct_1 += pred.eq(target.cuda().view_as(pred)).sum().item()

        total_train_loss_1 /= length
        total_correct_sum_1 += total_correct_1
        total_classification_loss_1 += total_train_loss_1
        print('epoch [{}/{}], loss-1:{:.4f} Accuracy-1: {}/{}'.format(epoch + 1, num_epochs, total_train_loss_1, total_correct_1, length))

        total_train_loss_2 = 0
        total_correct_2 = 0

        for data in train_data_loader:
            img, target = data
            img = Variable(img.view(img.size(0), -1)).cuda()

            optimizer_dense.zero_grad()

            output, features = model.features(img)

            loss = F.nll_loss(output, target.cuda())
            loss.backward()
            optimizer_dense.step()

            total_train_loss_2 += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total_correct_2 += pred.eq(target.cuda().view_as(pred)).sum().item()

        total_train_loss_2 /= length
        total_correct_sum_2 += total_correct_2
        total_classification_loss_2 += total_train_loss_2
        print('epoch [{}/{}], loss-2:{:.4f} Accuracy-2: {}/{}'.format(epoch + 1, num_epochs, total_train_loss_2,
                                                                      total_correct_2, length))

        total_train_loss_3 = 0
        total_correct_3 = 0
        for data in train_data_loader:
            img, target = data
            img = Variable(img.view(img.size(0), -1)).cuda()

            optimizer_whole.zero_grad()

            output, features = model.features(img)

            loss = F.nll_loss(output, target.cuda())
            loss.backward()
            optimizer_whole.step()

            total_train_loss_3 += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total_correct_3 += pred.eq(target.cuda().view_as(pred)).sum().item()

        total_train_loss_3 /= length
        total_correct_sum_3 += total_correct_3
        total_classification_loss_3 += total_train_loss_3
        print('epoch [{}/{}], loss-3:{:.4f} Accuracy-3: {}/{}'.format(epoch + 1, num_epochs, total_train_loss_3,
                                                                      total_correct_3, length))

        # final_test = test_train()
        # final_ref = test_train_ref()
        # if final_test > final_ref:
        #     print("Updating Ref: epoch: {}, accuracy: {} to {}".format(epoch, final_ref, final_test))
        ref_model.load_state_dict(model.state_dict())

    print("average correct 1:", total_correct_sum_1 / num_epochs)
    print("average loss 1:", total_classification_loss_1 / num_epochs)

    print("average correct 2:", total_correct_sum_2 / num_epochs)
    print("average loss 2:", total_classification_loss_2 / num_epochs)

    print("average correct 3:", total_correct_sum_3 / num_epochs)
    print("average loss 3:", total_classification_loss_3 / num_epochs)

    print("average correct:", total_correct_sum_1 + total_correct_sum_2 + total_correct_sum_3 / num_epochs / 3)
    print("average loss:", total_classification_loss_1 + total_classification_loss_2 + total_classification_loss_3 / num_epochs)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            data = data.view(data.size(0), -1).cuda()
            output = model(data)

            test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct += pred.eq(target.cuda()).sum().item()

    test_loss /= len(test_data_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))


def test_train():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_data_loader:
            data = data.view(data.size(0), -1).cuda()
            output = model(data)

            test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct += pred.eq(target.cuda()).sum().item()

    test_loss /= len(train_data_loader.dataset)

    # print('Test train set model: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(train_data_loader.dataset),
    #     100. * correct / len(train_data_loader.dataset)))
    return correct


def test_train_ref():
    ref_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_data_loader:
            data = data.view(data.size(0), -1).cuda()
            output = ref_model(data)

            test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct += pred.eq(target.cuda()).sum().item()

    test_loss /= len(train_data_loader.dataset)

    # print('Test train set reference: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(train_data_loader.dataset),
    #     100. * correct / len(train_data_loader.dataset)))
    return correct
# 1000, 500, 200, 100, 75, 50, 25, 10, 5, 1, 0.5,


if __name__ == '__main__':
    for j in [50, 25, 10, 5, 1, 0.5, 0.1, 0.05, 0.01]:
        alpha = j
        print(alpha)
        ref_model.load_state_dict(state_dict)
        model.load_state_dict(state_dict)
        residual_train()
        # loc_1 = "./CNN-New/layer-123-" + str(j) + ".pt"
        # torch.save(model.state_dict(), loc_1)
        # loc_2 = "./CNN-New/ref-layer-123-" + str(j) + ".pt"
        # torch.save(ref_model.state_dict(), loc_2)
        print(alpha)
