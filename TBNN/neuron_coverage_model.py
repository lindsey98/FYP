import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

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

        self.forward_map = dict()

        def forward_hook_fn(module, input, output):
            zero = torch.zeros_like(input[0], dtype=torch.int)
            one = torch.ones_like(input[0], dtype=torch.int)
            new_input = torch.where(input[0] > 0, one, zero)
            self.forward_map[module.name] = new_input

        for name, top_module in self.model.named_children():
            for idx, module in top_module._modules.items():
                if module.__class__.__name__ == 'ReLU':
                    coverage_relu = NeuronCoverageReLU(name).to(device)
                    coverage_relu.register_forward_hook(forward_hook_fn)
                    top_module._modules[idx] = coverage_relu

    def forward(self, x):
        return self.model.forward(x), self.forward_map

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
