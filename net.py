import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from .shuffle import shuffle


class AQPNet(nn.Module):

    def __init__(self, attr_num):
        super(AQPNet, self).__init__()
        self.attr_num = attr_num
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, padding_mode='circular')
        self.fc1 = nn.Linear(math.ceil(attr_num / 2) * 32, 128)
        self.fc2 = nn.Linear(128, 1)
        self.do1 = nn.Dropout(0.5)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2, ceil_mode=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.do1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.do1(x)
        return F.sigmoid(x)
