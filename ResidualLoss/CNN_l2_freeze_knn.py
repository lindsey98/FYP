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
evaluation_batch_size = 10000
learning_rate = 0.0001
alpha = 0.05
k = 1

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
    total_correct_sum = 0
    total_classification_loss = 0

    internal_signals_dict = dict()
    for i in range(10):
        internal_signals_dict[i] = list()

    with torch.no_grad():
        start_index = 0
        for data, target in evaluation_data_loader:
            data, target = data.cuda(), target.cuda()

            output, features = ref_model.features(data, retain_grad=False)
            resize_feature = features[2].view(features[2].size(0), features[2].size(1), -1).mean(axis=2)

            pred = output.argmax(dim=1)
            correct_list = pred.eq(target)

            for i in range(evaluation_batch_size):
                if correct_list[i].item():
                    internal_signals_dict[pred[i].item()].append(resize_feature[i])

            start_index += evaluation_batch_size

    for idx in internal_signals_dict:
        internal_signals_dict[idx] = torch.stack(internal_signals_dict[idx], dim=0)

    for epoch in range(num_epochs):
        total_correct = 0

        distance = -torch.ones(train_data_length, dtype=torch.float64)
        ones = torch.ones(train_data_length, dtype=torch.float64)

        model.eval()

        with torch.no_grad():
            start_index = 0
            for data, target in evaluation_data_loader:
                data, target = data.cuda(), target.cuda()

                output, features = model.features(data, retain_grad=False)
                resize_feature = features[2].view(features[2].size(0), features[2].size(1), -1).mean(axis=2)

                pred = output.argmax(dim=1)
                correct_list = pred.eq(target)
                total_correct += correct_list.sum().item()

                for i in range(evaluation_batch_size):
                    if not correct_list[i].item():
                        current_internal_signals = internal_signals_dict[target[i].item()]

                        dist = current_internal_signals.add(-resize_feature[i].expand(current_internal_signals.shape)).pow(2).sum(dim=1).pow(.5)
                        knn_indices = dist.topk(k, largest=False, sorted=False)
                        distance[start_index + i] = knn_indices[0].mean().item()

                start_index += evaluation_batch_size

        distance /= distance.max()
        score = torch.where(distance <= 0, ones, distance)
        probs = -torch.log(score)

        sampler.weights = probs + 1

        model.train()
        total_train_loss = 0

        for data, target in train_data_loader:
            data, target = data.cuda(), target.cuda()

            data = Variable(data)

            optimizer.zero_grad()

            output, features = model.features(data)

            ref_output, ref_features = ref_model.features(data)
            ref_pred = ref_output.argmax(dim=1)
            ref_list = ref_pred.eq(target).int()

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
        if epoch % 50 == 0:
            print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, train_data_length))
            test()
        # ref_model.load_state_dict(model.state_dict())

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
    for j in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        k = j
        print(k)
        ref_model.load_state_dict(state_dict)
        model.load_state_dict(state_dict)
        residual_train()
        loc = "./CNN-l2-freeze-knn/beta-" + str(k) + ".pt"
        torch.save(model.state_dict(), loc)
        print(k)
