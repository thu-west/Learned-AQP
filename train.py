import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from net import AQPNet
from shuffle import shuffle_batch
from einops import repeat
from globals import *
# We assume the attribute is labeled from 0 or we need normalize
# We have 10 attributes base on Professor Wang Ying


def train(model, device, data, target, optimizer, batch_size):
    model.train()
    loss_array = []
    for i in range(len(target) // batch_size):
        batch_data = data[i*batch_size:(i+1)*batch_size]
        batch_target = target[i*batch_size:(i+1)*batch_size]
        batch_data = torch.from_numpy(batch_data).to(device=device, dtype=torch.float)
        batch_target = torch.from_numpy(batch_target).to(device=device, dtype=torch.float)
        # In fact, we consider the input as the collection of 2D graphs, which is already 3D tensor
        # However, when the Conv2D requires the following format of input [batch, channels, width, height]
        # We only have one channel, such we should add extra dim here.
        # We use einops optimize the format to replace the follwing code
        # batch_data_size = list(batch_data.size())
        # batch_data_size.insert(1, 1)
        # batch_data = torch.reshape(batch_data, batch_data_size)
        batch_data = repeat(batch_data, 'b w h -> b c w h', c=1)

        optimizer.zero_grad()
        output = model(batch_data)
        loss = F.l1_loss(output, batch_target)
        loss.backward()
        optimizer.step()
        loss_array.append(loss.item())
    return loss_array


def process_train_set(train_sets, attr_num, shuffle_time):
    train_sets = np.array(train_sets)
    targets = train_sets[:, 0:1]
    datas = train_sets[:, 1:]
    datas = np.array(shuffle_batch(datas, attr_num, shuffle_time))
    return datas, targets


def main():
    device = torch.device("cuda")
    net = AQPNet(ATTR_NUM, SHUFFLE_TIME)
    model = net.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    # We Put Our Example Train Sets in Global Variables
    # And we provide the formats of the train sets
    data, target = process_train_set(EG_TRAIN_SETS, ATTR_NUM, SHUFFLE_TIME)
    batch_size = min(len(target), 32)
    for epoch in range(1, 100):
        train(model, device, data, target, optimizer, batch_size)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()
