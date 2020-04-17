import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import pandas as pd
from .net import AQPNet
from .shuffle import shuffle
from .globals import *
# We assume the attribute is labeled from 0 or we need normalize
# We have 10 attributes base on Professor Wang Ying


def train(model, device, data, target, optimizer, batch_size):
    model.train()
    loss_array = []
    for i in range(len(target) // batch_size):
        batch_data = data[i*batch_size:(i+1)*batch_size]
        batch_target = target[i*batch_size:(i+1)*batch_size]
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        optimizer.zero_grad()
        output = model(batch_data)
        loss = F.l1_loss(output, batch_target)
        loss.backward()
        optimizer.step()
        loss_array.append(loss.item())
    return loss_array


def process_train_set(train_sets):
    train_sets = np.array(train_sets)
    targets = train_sets[:, 0:1].T[0]
    datas = train_sets[:, 1:]
    datas = np.array(shuffle_batch(datas))
    return datas, targets


def main():
    device = torch.device("cuda")
    net = AQPNet(ATTR_NUM*ATTR_NUM)
    model = net.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    # We Put Our Example Train Sets in Global Variables
    # And we provide the formats of the train sets
    data, target = process_train_set(EG_TRAIN_SETS)
    batch_size = min(len(target), 32)
    for epoch in range(1, 100):
        train(model, device, data, target, optimizer, batch_size)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
