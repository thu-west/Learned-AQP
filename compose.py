# The Whole Idea is explained as follows
# For instance, Our aim is converting the query Q 0.1<=a<0.2 and 0.4<=b<0.6 to
# the combinations of the following 4 simple queries
# 0) -inf < a < 0.2 and -inf < b < 0.6  00 a use max, b use max, sign is +
# 1) -inf < a < 0.2 and -inf < b < 0.4  01 a use max, b use min, sign is -
# 2) -inf < a < 0.1 and -inf < b < 0.6  10 a use min, b use max, sign is -
# 3) -inf < a < 0.1 and -inf < b < 0.4  11 a use min, b use min, sign is +
# and the result of Q is 0) - 1) - 2) + 3)
# We find the number of simple queries is 2**attr_num
# And their formulas are related to the 0 or 1 in the position of binary number
# while the signs are related to the number of 1s the binary number
# Therefore we have the following algorithms for decompose and compose


import numpy as np


def convert_int_to_bool_list(i, attr_num=10):
    return [int(ch) for ch in list(format(i, 'b').zfill(attr_num))]


def decompose(input_query, attr_num=10):
    # query is a 2-d array 10 x 2
    # each row represents the attr while column represents the min and max
    # the query is already normalized to [0,1]
    # while the attribute is missing, we can give the related range [0,1]
    # input_query = np.array([[0.2, 0.3], [0.4, 0.5], [0.3, 0.9], [0.1, 0.7], [0.7, 1.0],
    #                  [0.0, 1.0], [0.1, 0.8], [0.7, 0.8], [0.2, 0.6], [0.4, 0.6]])
    output_query_list = []
    for i in range(0, 2**attr_num):
        binary_list = convert_int_to_bool_list(i)
        output_query = []
        for j in range(0, attr_num):
            # Since i = 0 represents all max
            # while i = 2**n -1 represents all min
            if binary_list[j] == 0:
                output_query.append(input_query[j][1])
            else:
                output_query.append(input_query[j][0])
        output_query_list.append(output_query)
    return np.array(output_query_list)


def compose(input_res, attr_num=10):
    # the input result are 2**attr_num and each is related to the above generated queries
    # input_res = np.random.random(2**attr_num)
    res = 0.0
    for i in range(0, 2**attr_num):
        count_one = np.sum(convert_int_to_bool_list(i))
        sign = (-1)**count_one
        res = res + sign * input_res[i]
    # Due to precision error, sometimes res while be small negative, we fix it by max
    return max(res, 0)
