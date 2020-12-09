import random
from torch.autograd import Variable
from torch import optim

import numpy as np
from torch.backends import cudnn
import torch.nn.functional as F
import torch

from ResidualLoss.dataset import cifar10_data_loader_test, cifar10_data_loader_train
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
learning_rate = 0.0001
alpha = 0.05
top_k = 4

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

train_data_loader = cifar10_data_loader_train(batch_size)
test_data_loader = cifar10_data_loader_test(batch_size)


def residual_train():
    total_correct_sum = 0
    total_classification_loss = 0
    length = len(train_data_loader.dataset)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_correct = 0
        for data in train_data_loader:
            img, target = data
            img, target = img.cuda(), target.cuda()
            img = Variable(img.view(img.size(0), -1))

            optimizer.zero_grad()
            output, features = model.features(img)
            pred = output.argmax(dim=1, keepdim=True)

            ref_output, ref_features = ref_model.features(img)
            ref_pred = ref_output.argmax(dim=1)

            # use the softmax instead of log_softmax here as design
            ref_softmax = F.softmax(ref_features[5], dim=1)
            top_k_softmax = ref_softmax.topk(k=top_k, dim=1, sorted=True)

            # if the sample is correct, pick top k element to calculate contribution
            lst_for_correct = []
            for k in range(top_k):
                # clear the gradient in the previous iteration
                ref_model.zero_grad()
                ref_features[2].grad = None

                # use sum(dim=0) because the loss is mutually exclusive,
                # this saves the effort to loop through each sample once
                top_k_softmax[0].sum(dim=0)[k].backward(retain_graph=True)
                lst_for_correct.append(
                    ref_features[2].grad
                        .detach().clone()
                        .view(features[2].size(0), features[2].size(1), -1)
                        .mean(axis=2)  # sum the gradient to simulate the gradient for GAP
                )

            # clear the gradient in the previous process
            ref_model.zero_grad()
            ref_features[2].grad = None

            # if the sample is wrong, pick the correct position to calculate contribution
            ref_softmax.gather(dim=1, index=target.unsqueeze(1)).sum().backward(retain_graph=True)
            value_for_wrong = ref_features[2].grad.detach().clone().view(features[2].size(0), features[2].size(1), -1).mean(axis=2)

            ref_correct_list = ref_pred.eq(target)

            new_loss = 0
            resize_feature = features[2].view(features[2].size(0), features[2].size(1), -1).mean(axis=2)
            resize_ref_feature = ref_features[2].view(ref_features[2].size(0), ref_features[2].size(1), -1).mean(axis=2)

            normalize_ref_feature = F.normalize(resize_ref_feature, dim=1).detach()
            normalize_feature = F.normalize(resize_feature, dim=1)
            feature_diff = normalize_feature - normalize_ref_feature

            for i in range(batch_size):
                if ref_correct_list[i].item():
                    temp_yita = 0
                    count = 0
                    for k in range(top_k - 1):
                        temp_yita += lst_for_correct[k + 1][i]
                        count += 1

                    yita = lst_for_correct[0][i] - temp_yita / count
                    temp_loss = torch.norm(feature_diff[i] * yita, p=1)
                    new_loss -= temp_loss
                else:
                    temp_yita = 0
                    count = 0
                    for k in range(top_k):
                        if top_k_softmax[1][i][k].item() == target[i].item() or count == top_k - 1:
                            continue
                        else:
                            temp_yita += lst_for_correct[k][i]
                            count += 1

                    yita = value_for_wrong[i] - temp_yita / count
                    temp_loss = torch.norm(feature_diff[i] * yita, p=1)
                    new_loss -= temp_loss

            loss = F.nll_loss(output, target, reduction='mean')
            loss += alpha * new_loss

            loss.backward()
            optimizer.step()

            total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            total_correct += pred.eq(target.view_as(pred)).sum().item()

        total_train_loss /= length
        total_correct_sum += total_correct
        total_classification_loss += total_train_loss
        if epoch % 50 == 0:
            print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, length))
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
            data = data.view(data.size(0), -1)
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
        ref_model.load_state_dict(state_dict)
        model.load_state_dict(state_dict)
        residual_train()
        loc = "./CNN-l2-freeze-new/kk2-" + str(j) + ".pt"
        torch.save(model.state_dict(), loc)
        print(alpha)
