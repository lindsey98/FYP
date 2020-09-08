from torch.autograd import Variable
from torch import optim, save, load

import torch.nn.functional as F
import torch

from NeuronCoverage.dataset import minst_data_loader_test, minst_data_loader_train
from NeuronCoverage.model import PaperModel, MINST_3
from NeuronCoverage.neuron_coverage_model import NeuronCoverageReLUModel

num_epochs = 15
batch_size = 100
learning_rate = 0.0001

model = MINST_3().cuda()

model = NeuronCoverageReLUModel(model)

optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

train_data_loader = minst_data_loader_train(batch_size)
test_data_loader = minst_data_loader_test(batch_size)


def train():
    model.load_state_dict(load('./states/MINST-3.pth'))
    model.train()
    length = len(train_data_loader.dataset)
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_correct = 0
        for data in train_data_loader:
            img, target = data
            img = Variable(img.view(img.size(0), -1)).cuda()

            optimizer.zero_grad()
            output = model.forward(img)

            loss = F.nll_loss(output, target.cuda())
            loss.backward()
            optimizer.step()

            total_train_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total_correct += pred.eq(target.cuda().view_as(pred)).sum().item()

        total_train_loss /= length
        print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, length))
    save(model.state_dict(), './states/MINST-3-1.pth')


def test():
    model.load_state_dict(load('./states/MINST-3-1.pth'))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            data = data.view(data.size(0), -1).cuda()
            output = model(data)

            test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.cuda().view_as(pred)).sum().item()

    test_loss /= len(test_data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))


if __name__ == '__main__':
    train()
    test()
