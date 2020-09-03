import random

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.backends import cudnn

from TBNN.dataset import minst_data_loader_train, minst_data_loader_test
from TBNN.model import MINST_3, MINST_8, MINST_9
from TBNN.neuron_coverage_model import NeuronCoverageReLUModel


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


setup_seed(1914)
batch_size = 128
num_epochs = 200
learning_rate = 0.0001
model = NeuronCoverageReLUModel(MINST_3())
# model.load_state_dict(torch.load('./states/MINST-3.pth'))

train_data_loader = minst_data_loader_train(batch_size)
test_data_loader = minst_data_loader_test(batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

f = open("record1.txt", "w")


if __name__ == '__main__':
    model = NeuronCoverageReLUModel(MINST_9())
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.combined_train(20, train_data_loader, optimizer, log_address="./log/record3.txt")
    model.normal_test(test_data_loader, load_from="./states/MINST-3-11.pth")
