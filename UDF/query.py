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
    # print("output_queries:", output_queries.shape)
    # print(output_queries)
    shuffle_output_queries = shuffle_batch(output_queries, ATTR_NUM, SHUFFLE_TIME)
    tensor_queries = torch.from_numpy(np.array(shuffle_output_queries)).to(device=device, dtype=torch.float)
    # print("tensor_queries shape:", tensor_queries.shape)
    # Same Reason in train.py
    # We use einops optimize the format to replace the follwing code
    # batch_data_size = list(batch_data.size())
    # batch_data_size.insert(1, 1)
    # batch_data = torch.reshape(batch_data, batch_data_size)
    tensor_queries = repeat(tensor_queries, 'b w h -> b c w h', c=1)
    output_tensors = model(tensor_queries)
    # print("tensor_queries:",tensor_queries.shape)
    # print("output_tensors:",output_tensors.shape)
    output_array = output_tensors.data.cpu().numpy()
    # print("before",output_array.shape)
    output_array = np.reshape(output_array, output_array.size)
    # print("after",output_array.shape)
    
    # store trained model as a torchscript file
    traced_script_module = torch.jit.trace(model, tensor_queries)
    output = traced_script_module(tensor_queries)
    traced_script_module.save("./modelscript.pt")
    # print("check:",torch.sum(output != output_tensors))

    res = compose(output_array)
    return res

# def main(args):
#     example query
#     query = np.array(args[1:])

def main():
    query = np.array([[0.0, 0.8], [0.0, 0.8], [0.0, 0.8], [0.5, 0.9], [0.2, 1.0],
                      [0.2, 0.8], [0.5, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    # print(query.shape)
    res = do_query(query)
    print("res: ", res[0] * 100, "%")
    # The res here are the percentage of estimated number in the whole number



if __name__ == '__main__':
    main()
