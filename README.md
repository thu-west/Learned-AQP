Code Contributors: 
- Yang Duan yangd4@illinois.edu for UDF Implementations
- Jiacheng Wu wjcskqygj@gmail.com for Python Implementation

If you have any questions, feel free to contact us either by creating an issue on Github or sending emails. 

Please cite our paper if you choose to use our code.
```bash
@article{
  title={Database Native Approximate Query Processing Based on Machine-Learning},
  author={Yang Duan, Yong Zhang, Jiacheng Wu},
  booktitle={TBD},
  year={2021},
}
```

# Introduction

The goal of this project is to design a new AQP module based on Machine Learning and embed it inside databases.     
With the help of User Defined Functions, we successfully embedded it inside MySQL Database.          
More details are in our paper [TBD paper](https://TBD).      
   
### Table of Contents
**[Dependency](#Dependency)**<br>
**[Installation and Usage](#Installation-and-Usage)**<br>
**[Format of the training dataset](#Format-of-the-training-dataset)**<br>
**[File Descriptions](#File-Descriptions)**<br>
**[Algorithm Overview](#Algorithm-Overview)**<br>


# Dependency
We tested these implementations on Windows Subsystem for Linux 2 (WSL2) with Ubuntu 18.04

## Python implementation
python 3 + torch + numpy + einops

```bash
pip install torch numpy einops 
```

## UDF implementations
In addition to these mentioned packages, we also use the following specified versions of software that correspond to our hardware:     
- CMake 3.19.2
- GCC 7.5.0
- Pytorch Stable 1.8.1
- Cudatoolkit 11.0 on WSL2
- Cudnn 8.2.0
- Boost 1.69.0
- Numcpp 2.4.2
- lib_mysqludf_sys

# Installation and Usage

## Python implementation

1. There is one example training set which is an array in **globals.py**.  You can use numpy to load a training set if it is in other formats, like csv.
2. After loading the training set, you can run **train.py** to start training the model. The trained model will be saved to **MODEL_SAVE_PATH** in **globals.py**.
3. In order to answer queries, we need to run **query.py**. There is one example query in **query.py**. The output in stdout is the estimated result of the query.

## UDF implementations   
All of the following operations need to be done in Folder UDF. 
1. We need to first do one query by the mentioned Python implementation, the one in Folder UDF, to obtain the trained model. 
2. After running **query.py**, the torchscript **modelscript.pt** will be saved in the current path. Then, it needs to be copied to the path of mysql.
```bash
sudo cp modelscript.pt /etc/mysql/modelscript.pt
sudo chmod 777 /etc/mysql/modelscript.pt
```
3. You need to modify **CMakeLists.txt** so that the paths of packages in it are the paths of installed packages in your computer.

### ETIQ UDF
4. run the following linux commands
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo cp libmyAQP.so /usr/lib/mysql/plugin/
```
5. run the following commands in MySQL
```bash
DROP FUNCTION IF EXISTS myAQP;
CREATE FUNCTION myAQP RETURNS REAL SONAME 'libmyAQP.so';
```
6. Usage
```bash
select myAQP(<lower bound of Attribute 1>,<upper bound of Attribute 1>, ... , <lower bound of Attribute n>, <upper bound of Attribute n>);
```
7. an example for datasets with 10 attributes
```bash
select myAQP(0.2,0.52,0.1,0.23,0.065,0.27,0.055,0.87,0.32,0.68,0.01,0.78,0.27,0.83,0.005,0.35,0.03,0.08,0.46,0.99);
```
### ETEQ UDF
4. comment the codes of ETIQ in CMakeLists.txt and uncomment the codes of ETEQ
5. run the following linux commands
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
6. Usage
```bash
select sys_eval('<path of ETEQ executable> <lower bound of Attribute 1>  <upper bound of Attribute 1> ... <lower bound of Attribute n> <upper bound of Attribute n>');
```
7. an example for datasets with 10 attributes
```bash
select sys_eval('/mnt/c/Learned-AQP/UDF/build/ETEQ 0.025 0.9 0.32 0.64 0.15 0.87 0.035 0.46 0.23 0.65 0.19 0.87 0.4 0.96 0.32 0.97 0.065 0.98 0.05 0.33');
```

# Format of the training dataset

It should be a 2d array with n rows and 11 columns if the dataset has ten attributes. In this example, the col0 represents the percentage of data points satisfying attr1 < col1 and attr2 < col2 and ... and attr10 < col10, and all of these attributes are normalized within the range [0,1].


# File Descriptions

1. globals.py: defines some global variables and the training dataset
2. shuffle.py: defines functions that do permutations for the input array to help us correctly utilize Convolutional Neural Network later by reducing the distances of different attributes. If we have ten attributes, it will do permutations and stack them vertically to obtain one 10*10 2d image as the new input to CNN.
3. net.py: defines the Convolutional Neural Network with simple models to prevent overfitting
4. train.py: defines the training process with PyTorch
5. compose.py: defines functions that compose and decompose the input array with the algorithm which we mentioned in our paper
6. query.py: defines functions that do queries; saves models for UDF implementations (the version of query.py in Folder UDF)
7. UDF/CMakeLists.txt: the CMake file to compile UDF implementations
8. UDF/ETEQ.cpp: codes of ETEQ UDF
9. UDF/myAQP.cpp: codes of ETIQ UDF
10. UDF/aqp_TPCH.pt: the trained model in our paper for the TPCH dataset
11. UDF/aqp_synthetic.pt: the trained model in our paper for the synthetic dataset
12. UDF/modelscript_TPCH.pt: the torchscript of the trained model in our paper for the TPCH dataset
13. UDF/modelscript_synthetic.pt: the torchscript of the trained model in our paper for the synthetic dataset

# Algorithm Overview

Let's take the dataset with three attributes as an example (we use the example of a dataset with two attributes in the comments of our codes).     

We define a query of [0, A1 ] ^ [0, A2] ^ [0, A3] where Ai is the upper bound of Attribute i as Simple Query. The results of Simple Query can be computed by Convolutional Neural Network.       

In the same way, we define a query of [S1, E1] ^ [S2, E2] ^ [S3, E3] where Si is the lower bound of Attribute i and Ei is the upper bound of Attribute i as Complex Query.    

By Principle of inclusion-exclusion, Complex Query is the combination of many Simple Query, which we explained in detail in our paper [TBD paper](https://TBD).    

For this example, it can generate 8 = 2^3 combinations since every attribute has one lower bound and one upper bound in every query，  

0. 0b000 => R0 = [0, E1] ^ [0, E2] ^ [0, E3]
1. 0b001 => R1 = [0, E1] ^ [0, E2] ^ [0, S3]
2. 0b010 => R2 = [0, E1] ^ [0, S2] ^ [0, E3]
3. 0b011 => R3 = [0, E1] ^ [0, S2] ^ [0, S3]
4. 0b100 => R4 = [0, S1] ^ [0, E2] ^ [0, E3]
5. 0b101 => R5 = [0, S1] ^ [0, E2] ^ [0, S3]
6. 0b110 => R6 = [0, S1] ^ [0, S2] ^ [0, E3]
7. 0b111 => R7 = [0, S1] ^ [0, S2] ^ [0, S3]

By Principle of inclusion-exclusion, we can see R = R0 - R1 - R2 - R3 + R4 + R5 + R6 - R7        

In our implementations, we loop from 0 to 2^n-1. In other words, there are 2^n numbers in total. Based on bits of them in the binary format, we choose the lower bound or upper bound. For example, Number 5 is 0b101 in binary. The first bit is 1, so S1 should be the upper bound for Attribute 1 in this combination. The second bit is 0, so E2 should be the upper bound for Attribute 2 in this combination, and so on. In this way, we produce Combination R5.     

With the results of corresponding queries of these 2^n combinations from CNN, we need to do arithmetic operations to get the final answer of the Complex Query. In order to do so, we first calculate the number of Bit 1 of them in binary format. If the number is even, we add the corresponding result; if the number is odd, we subtract them. In the above example, 5 in binary format is 0b101. Since it has two Bit 1, which is even, we add the corresponding result to our final answer.

