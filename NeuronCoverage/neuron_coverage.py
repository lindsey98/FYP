import random
import time

import torch
import numpy as np
from torch.backends import cudnn

from NeuronCoverage.dataset import minst_data_loader_train, minst_data_loader_test, fasion_minst_data_loader_train, \
    fasion_minst_data_loader_test
from NeuronCoverage.model import MINST_3, MINST_8, MINST_9
from NeuronCoverage.neuron_coverage_model import NeuronCoverageReLUModel


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

train_data_loader = fasion_minst_data_loader_train(batch_size)
test_data_loader = fasion_minst_data_loader_test(batch_size)


if __name__ == '__main__':
    model = NeuronCoverageReLUModel(MINST_9())
    model2 = NeuronCoverageReLUModel(MINST_9())
    model2.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-5)
    for i in range(30):
        print(i)
        print("Combined")
        model.combined_train(5, train_data_loader, optimizer, log_address="./log/FMINST-9-1.txt", load_from='./states/FMINST-9-1-%d.pth')
        model.normal_test(test_data_loader)

