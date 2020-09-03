import torch
from torch import nn

import torch.nn.functional as F


class PaperModel(nn.Module):
    def __init__(self):
        super(PaperModel, self).__init__()
        self.hidden1 = nn.Linear(784, 784)
        self.hidden2 = nn.Linear(784, 784)
        self.output = nn.Linear(784, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_1(nn.Module):
    def __init__(self):
        super(MINST_1, self).__init__()
        self.output = nn.Linear(784, 10)

    def forward(self, x):
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_2(nn.Module):
    def __init__(self):
        super(MINST_2, self).__init__()
        self.hidden1 = nn.Linear(784, 200)
        self.hidden2 = nn.Linear(200, 100)
        self.hidden3 = nn.Linear(100, 60)
        self.hidden4 = nn.Linear(60, 30)
        self.hidden5 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = torch.sigmoid(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.hidden3(x)
        x = torch.sigmoid(x)
        x = self.hidden4(x)
        x = torch.sigmoid(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_3(nn.Module):
    def __init__(self):
        super(MINST_3, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(100, 60),
            nn.ReLU(inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(60, 30),
            nn.ReLU(inplace=True)
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_4(nn.Module):
    def __init__(self):
        super(MINST_4, self).__init__()
        self.hidden1 = nn.Linear(784, 200)
        self.hidden2 = nn.Linear(200, 100)
        self.hidden3 = nn.Linear(100, 60)
        self.hidden4 = nn.Linear(60, 30)
        self.hidden5 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.hidden2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.hidden3(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.hidden4(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_7(nn.Module):
    def __init__(self):
        super(MINST_7, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(784, 200),
            nn.BatchNorm1d(200),
            nn.Sigmoid()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.Sigmoid()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(100, 60),
            nn.BatchNorm1d(60),
            nn.Sigmoid()
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(60, 30),
            nn.BatchNorm1d(30),
            nn.Sigmoid()
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_8(nn.Module):
    def __init__(self):
        super(MINST_8, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(784, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(100, 60),
            nn.BatchNorm1d(60),
            nn.ReLU(inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(60, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True)
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_9(nn.Module):
    def __init__(self):
        super(MINST_9, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x
