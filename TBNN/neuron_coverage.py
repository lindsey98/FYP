import torch
from torch.autograd import Variable
import torch.nn.functional as F

from TBNN.dataset import minst_data_loader_train, minst_data_loader_test
from TBNN.model import MINST_3, MINST_8
from TBNN.neuron_coverage_model import NeuronCoverageReLUModel

batch_size = 128
num_epochs = 200
learning_rate = 0.0001
model = NeuronCoverageReLUModel(MINST_3())
model.load_state_dict(torch.load('./states/MINST-3.pth'))

train_data_loader = minst_data_loader_train(batch_size)
test_data_loader = minst_data_loader_test(batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


def coverage():
    model.coverage()
    test_loss = 0
    correct = 0

    neuron_activation_map_correct = dict()
    neuron_activation_map_wrong = dict()
    with torch.no_grad():
        for data, target in train_data_loader:
            data = data.view(data.size(0), -1).cuda()
            output = model.forward(data)
            test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.cuda().view_as(pred)
            correct_pred = pred.eq(target).int()

            for name, neuron_activation in model.neuron_activation_map.items():
                pred_neuron_activation_correct = (neuron_activation * correct_pred).sum(axis=0)
                pred_neuron_activation_wrong = (neuron_activation * (1 - correct_pred)).sum(axis=0)

                if neuron_activation_map_correct.get(name) is None:
                    neuron_activation_map_correct[name] = pred_neuron_activation_correct
                else:
                    neuron_activation_map_correct[name] += pred_neuron_activation_correct

                if neuron_activation_map_wrong.get(name) is None:
                    neuron_activation_map_wrong[name] = pred_neuron_activation_wrong
                else:
                    neuron_activation_map_wrong[name] += pred_neuron_activation_wrong

            correct += correct_pred.sum().item()

    length = len(train_data_loader.dataset)
    test_loss /= length
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, length, 100. * correct / length))

    for name, activation_map_correct in neuron_activation_map_correct.items():
        activation_map_wrong = neuron_activation_map_wrong[name]

        contribution = (activation_map_correct.double() / correct) / ((activation_map_wrong + 1).double() / (length - correct))
        contribution_ratio = 1 - 1 / (contribution + 1)
        model.non_frozen_neuron_map[name] = (contribution_ratio < 0.5).int()


def freeze_train(epochs):
    model.freeze_train()
    length = len(train_data_loader.dataset)
    for epoch in range(epochs):
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
        print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, epochs, total_train_loss, total_correct, length))
    torch.save(model.state_dict(), './states/MINST-3-A.pth')


def test():
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

    length = len(test_data_loader.dataset)
    test_loss /= length
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, length, 100. * correct / length))


if __name__ == '__main__':
    coverage()
    freeze_train(10)
    test()
