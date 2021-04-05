import torch
import numpy as np
from einops import repeat
from net import AQPNet
from shuffle import shuffle_batch
from compose import compose, decompose
from globals import *


def load_model(device):
    net = AQPNet(ATTR_NUM, SHUFFLE_TIME)
    net.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model = net.to(device)
    model.eval()
    return model


def do_query(query):
    device = torch.device("cuda")
    model = load_model(device)
    output_queries = decompose(query, ATTR_NUM)
    # print(output_queries)
    shuffle_output_queries = shuffle_batch(output_queries, ATTR_NUM, SHUFFLE_TIME)
    tensor_queries = torch.from_numpy(np.array(shuffle_output_queries)).to(device=device, dtype=torch.float)
    # Same Reason in train.py
    tensor_queries = repeat(tensor_queries, 'b w h -> b c w h', c=1)
    output_tensors = model(tensor_queries)
    output_array = output_tensors.data.cpu().numpy()
    # output_array = np.reshape(output_array, output_array.size)
    res = compose(output_array)
    return res


def main():
    # example query
    query = np.array([[0.2, 0.3], [0.4, 0.5], [0.3, 0.9], [0.1, 0.7], [0.7, 1.0],
                      [0.0, 1.0], [0.1, 0.8], [0.7, 0.8], [0.2, 0.6], [0.4, 0.6]])
    res = do_query(query)
    print("res: ", res)
    # The res here are the percentage of estimated number in the whole number


if __name__ == '__main__':
    main()
