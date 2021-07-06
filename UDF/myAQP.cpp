#include <mysql/mysql.h>

#include <torch/torch.h>
#include <torch/script.h> 

#include "NumCpp.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <vector>

#define ATTR_NUM 10
#define SHUFFLE_TIME 10
#define MODEL_SAVE_PATH "/etc/mysql/modelscript.pt"

using namespace std;

// All of the following functions are translated from Python functions with the same names.
// The Function myAQP corresponds to Python Function do_query.

vector<int> convert_int_to_bool_list(int i, int attr_num=ATTR_NUM){
    // format(i, 'b')
    int input = i;
    bitset<64> bit(input);
    string str = bit.to_string();
    if(str.find("1") != str.npos){
        str = str.substr(str.find("1"));
    }else{
        str = "0";
    }

    // zfill(attr_num)
    string zeros;
    if(str.length() < attr_num){
        int diff = attr_num - str.length();
        for(int idx = 0; idx < diff; idx++){
            zeros += "0";
        }
        zeros += str;
    }else{
        zeros = str;
    }

    // [int(ch)]
    vector<int> result;
    for(int idx = 0; idx < zeros.length(); idx++){
        int element = zeros[idx] - '0';
        result.push_back(element);
    }
    return result;
}

nc::NdArray<double> decompose(nc::NdArray<double> input_query, int attr_num=ATTR_NUM){
    vector<vector<double>> output_query_list;
    int end = pow(2, attr_num);
    for(int i = 0; i < end; i++){
        vector<int> binary_list = convert_int_to_bool_list(i);
        vector<double> output_query;
        for(int j = 0; j < attr_num; j++){
            if(binary_list.at(j) == 0){
                output_query.push_back(input_query(j,1));
            }else{
                output_query.push_back(input_query(j,0));
            }
        }
        output_query_list.push_back(output_query);
    }
    nc::NdArray<double> result(output_query_list);
    return result;
}


nc::NdArray<double> shuffle(nc::NdArray<double> attr_data, int attr_num=ATTR_NUM, int shuffle_times=SHUFFLE_TIME){
    nc::random::seed(41);
    nc::NdArray<int> random_shuffle_seed = nc::random::permutation( nc::arange<int>(0, shuffle_times) );
    nc::NdArray<double> arrays = nc::copy(attr_data);
    
    nc::NdArray<double> tmp = nc::copy(attr_data);;
    for(int i = 1; i < shuffle_times; i ++){
        nc::random::seed(random_shuffle_seed(0,i));
        tmp = nc::random::permutation(tmp);
        arrays = nc::vstack({arrays, tmp});
    }
    return arrays;
}

vector<nc::NdArray<double>> shuffle_batch(nc::NdArray<double> batch, int attr_num=ATTR_NUM, int shuffle_times=SHUFFLE_TIME){
    vector<nc::NdArray<double>> result;	
    for(int i = 0; i < batch.shape().rows; i++){
        result.push_back(shuffle(batch(i, batch.cSlice()), attr_num, shuffle_times));
    }
    return result;
}

double compose(torch::Tensor input_res, int attr_num=ATTR_NUM){
    double res = 0.0;
    
    for(int i = 0; i < pow(2, attr_num); i++){
        vector<int> bool_list = convert_int_to_bool_list(i);
        double count_one = 0;
        for(int idx = 0; idx < bool_list.size(); idx++){
            count_one = count_one + bool_list.at(idx);
        }
        int sign = pow(-1, count_one);
        res = res + sign * input_res[i].item().toDouble();
    }
    return max(res, 0.0);
}

extern "C" double myAQP(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error)
{   
    // produce a numcpp array
    nc::NdArray<double> query = nc::zeros<double>(args->arg_count / 2, 2);

    for(int idx = 0; idx < args->arg_count; idx +=2){
        double lower = stod(args->args[idx]);
        double upper = stod(args->args[idx +1]);
        query((idx /2), 0) = lower;
        query((idx /2), 1) = upper;
    }

    // Load model
    string model_path = MODEL_SAVE_PATH;
    torch::jit::script::Module module = torch::jit::load(model_path);  

    nc::NdArray<double> output_queries = decompose(query, ATTR_NUM);
    vector<nc::NdArray<double>> shuffle_output_queries = shuffle_batch(output_queries, ATTR_NUM, SHUFFLE_TIME);
    
    vector<at::Tensor> tensors;
    for(int i = 0; i < shuffle_output_queries.size(); i++){
        nc::NdArray<double> currnc = shuffle_output_queries.at(i);
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        at::Tensor currTensor = torch::from_blob(currnc.data(), {currnc.numRows(), currnc.numCols()}, options).clone();
        tensors.push_back(currTensor.reshape({1, currnc.numRows(), currnc.numCols()}));
    }
    at::Tensor tensor_queries = torch::cat(tensors,0);
    tensor_queries = tensor_queries.unsqueeze(1);
    
    vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_queries);
    torch::Tensor output_tensors = module.forward(inputs).toTensor();
    output_tensors = output_tensors.reshape({-1});
    
    double res = compose(output_tensors);

    // convert to percentage
    return res * 100;
}

extern "C" my_bool myAQP_init(UDF_INIT *initid, UDF_ARGS *args, char *message)
{   
    initid->decimals = 10; 
    return 0;
}

extern "C" void myAQP_deinit(UDF_INIT *initid)
{   

}

void myAQP_add(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error)
{
    
}


void myAQP_clear(UDF_INIT *initid, char *is_null, char *error)
{

}