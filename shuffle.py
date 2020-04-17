import numpy as np

# We use the following idea to shuffle
# input = input[::2] + input[1::2]
# and repeat A = ceil(log2(attr_num)) times
# the get A * attr_num images
# Actually we can repeat more with more interesting shuffle
# The reason we try to do this shuffle is to make the connections more arbitrary
# Rather than simply use the locality from nearest attributes due to the Conv
# OR we can create/investigate more proper network to fit if possible

# input attr_data is like [0.2,0.4,..., 0.5]
def shuffle(attr_data, attr_num=10, shuffle_times=10):
    assert attr_num == shuffle_times
    assert len(attr_data) == attr_num
    # input = input[::2] + input[1::2]
    arrays = []
    random_shuffle_seed = np.random.RandomState(41).permutation(np.arange(0, shuffle_times))
    arrays.append(attr_data)
    tmp = attr_data
    for i in range(1, shuffle_times):  # 9 iteration
        tmp = np.random.RandomState(random_shuffle_seed[i]).permutation(tmp)
        arrays.append(tmp)
    return np.vstack(arrays)


def shuffle_batch(batch, attr_num=10, shuffle_times=10):
    return [shuffle(attr_data) for attr_data in batch]