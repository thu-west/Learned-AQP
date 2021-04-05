import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AQPNet(nn.Module):

    def __init__(self, attr_num, shuffle_time):
        super(AQPNet, self).__init__()
        first_level_pad = 2
        second_level_pad = 1
        total_pad = first_level_pad * second_level_pad
        assert attr_num % total_pad == 0
        assert shuffle_time % total_pad == 0
        self.conv1 = nn.Conv2d(1, 8, 5, padding=first_level_pad, padding_mode='circular')
        self.conv2 = nn.Conv2d(8, 16, 3, padding=second_level_pad, padding_mode='circular')
        # After convolution, the size of length and width would be deduced to 1 / total_pad of the origin
        # Thus, after flatten, we only have attr_num//total_pad * shuffle_time//total_pad * 16 vals
        self.fc1 = nn.Linear(attr_num//total_pad * shuffle_time//total_pad * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.do1 = nn.Dropout(0.5)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, ceil_mode=True)
        x = self.conv2(x)
        x = F.relu(x)
        # the flatten should not participate on the dim 0, which is batch dim
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        x = self.do1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.do1(x)
        return torch.sigmoid(x)
