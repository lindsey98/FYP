import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuronCoverageReLU(nn.Module):
    def __init__(self, name):
        super(NeuronCoverageReLU, self).__init__()
        self.name = name

    def forward(self, x):
        return F.relu(x)


class NeuronCoverageReLUModel:
    def __init__(self, model):
        self.model = model
        self.model = model.to(device)

        self.neuron_activation_map = dict()
        self.non_frozen_neuron_map = dict()
        self.mode = "normal"

        def forward_hook_fn(module, input, output):
            if self.mode == "coverage" or self.mode == "combined":
                zero = torch.zeros_like(input[0], dtype=torch.int)
                one = torch.ones_like(input[0], dtype=torch.int)
                new_input = torch.where(input[0] > 0, one, zero)
                self.neuron_activation_map[module.name] = new_input

        def backward_hook_fn(module, grad_in, grad_out):
            if self.mode == "freeze" or self.mode == "combined":
                mask = self.non_frozen_neuron_map.get(module.name)
                if mask is not None:
                    new_grad_in = grad_in[0] * mask
                    return (new_grad_in,)

        for name, top_module in self.model.named_children():
            for idx, module in top_module._modules.items():
                if module.__class__.__name__ == 'ReLU':
                    coverage_relu = NeuronCoverageReLU(name).to(device)
                    coverage_relu.register_forward_hook(forward_hook_fn)
                    coverage_relu.register_backward_hook(backward_hook_fn)
                    top_module._modules[idx] = coverage_relu

    def forward(self, x):
        return self.model.forward(x)

    def train(self):
        self.model.train()
        self.mode = "normal"

    def eval(self):
        self.model.eval()
        self.mode = "normal"

    def coverage(self):
        self.model.eval()
        self.mode = "coverage"

    def freeze_train(self):
        self.model.train()
        self.mode = "freeze"

    def combined(self):
        self.model.train()
        self.mode = "combined"

    def parameters(self):
        return self.model.parameters()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()

    def __call__(self, x):
        return self.forward(x)

    def normal_train(self, num_epochs, data_loader, optimizer, load_from=None, save_to=None, log_address="./log/default.txt"):
        self.train()
        length = len(data_loader.dataset)
        if load_from is not None:
            self.load_state_dict(torch.load(load_from))

        f = open(log_address, "a")

        for epoch in range(num_epochs):
            total_train_loss = 0
            total_correct = 0
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                data = Variable(data.view(data.size(0), -1))

                optimizer.zero_grad()
                output = self.forward(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total_correct += pred.eq(target.view_as(pred)).sum().item()

            total_train_loss /= length
            print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, length))
            f.write('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}\n'.format(epoch + 1, num_epochs, total_train_loss, total_correct, length))
            f.flush()
        f.close()
        if save_to is not None:
            torch.save(self.state_dict(), save_to)

    def combined_train(self, num_epochs, data_loader, optimizer, alpha=0.9, load_from=None, save_to=None, log_address="./log/default.txt"):
        self.combined()
        length = len(data_loader.dataset)
        if load_from is not None:
            self.load_state_dict(torch.load(load_from))

        f = open(log_address, "a")
        for epoch in range(num_epochs):
            total_train_loss = 0
            total_correct = 0

            neuron_activation_map_correct = dict()
            neuron_activation_map_wrong = dict()

            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                data = Variable(data.view(data.size(0), -1))

                optimizer.zero_grad()
                output = self.forward(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total_correct += pred.eq(target.view_as(pred)).sum().item()

                target = target.view_as(pred)
                correct_pred = pred.eq(target).int()

                for name, neuron_activation in self.neuron_activation_map.items():
                    shape = neuron_activation.shape[1:]
                    neuron_activation = neuron_activation.view(correct_pred.shape[0], -1)
                    pred_neuron_activation_correct = (neuron_activation * correct_pred).sum(axis=0).view(shape)
                    pred_neuron_activation_wrong = (neuron_activation * (1 - correct_pred)).sum(axis=0).view(shape)

                    if neuron_activation_map_correct.get(name) is None:
                        neuron_activation_map_correct[name] = pred_neuron_activation_correct
                    else:
                        neuron_activation_map_correct[name] += pred_neuron_activation_correct

                    if neuron_activation_map_wrong.get(name) is None:
                        neuron_activation_map_wrong[name] = pred_neuron_activation_wrong
                    else:
                        neuron_activation_map_wrong[name] += pred_neuron_activation_wrong

            total_train_loss /= length
            print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, length))
            f.write('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}\n'.format(epoch + 1, num_epochs, total_train_loss, total_correct, length))
            f.flush()
            for name, activation_map_correct in neuron_activation_map_correct.items():
                activation_map_wrong = neuron_activation_map_wrong[name]

                # contribution = (activation_map_correct.double() / total_correct) / (
                #         (activation_map_wrong + 1).double() / (length - total_correct))
                contribution = activation_map_correct.double() / (activation_map_wrong + 1).double()
                contribution_ratio = 1 - 1 / (contribution + 1)
                # print(contribution_ratio)
                self.non_frozen_neuron_map[name] = (contribution_ratio < alpha).int()

        f.close()
        if save_to is not None:
            torch.save(self.state_dict(), save_to)

    def normal_test(self, data_loader, load_from=None):
        self.eval()
        if load_from is not None:
            self.load_state_dict(torch.load(load_from))
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = self.forward(data)

                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        length = len(data_loader.dataset)
        test_loss /= length
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, length, 100. * correct / length))
