import torch
from torch import nn
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
