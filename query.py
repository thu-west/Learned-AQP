import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from .net import AQPNet
from .shuffle import shuffle, shuffle_batch
from .compose import compose, decompose
from .globals import *


def load_model():
    net = AQPNet(ATTR_NUM*ATTR_NUM)
    net.load_state_dict(torch.load(MODEL_SAVE_PATH))
    device = torch.device("cuda")
    model = net.to(device)
    model.eval()
    return model

def do_query(query):
    model = load_model()
    output_queries = decompose(query, ATTR_NUM)
    shuffle_output_queries = shuffle_batch(output_queries)
    tensor_queries = torch.from_numpy(np.array(shuffle_output_queries))
    output_tensors = model(tensor_queries)
    output_array = output_tensors.data.cpu().numpy()
    res = compose(output_array)

def main():
    # example query
    query = np.array([[0.2, 0.3], [0.4, 0.5], [0.3, 0.9], [0.1, 0.7], [0.7, 1.0],
                      [0.0, 1.0], [0.1, 0.8], [0.7, 0.8], [0.2, 0.6], [0.4, 0.6]])
    res = do_query(query)
    # The res here are the percentage of estimated number in the whole number
