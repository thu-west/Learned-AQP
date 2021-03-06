# Learning Based AQP

## 安装环境

python 3 + torch + numpy

```bash
pip install torch numpy
```

## 使用方法

1. 训练数据的示例在global.py里使用数组表示（待会会具体说明），其他类型数据可以使用csv格式，然后numpy导入，并不麻烦
2. 导入数据后，使用train.py训练即可，替换第46行的EG_TRAIN_SETS即可进行训练。训练完毕后，模型会导入到全局变量的MODEL_SAVE_PATH中。
3. 之后使用query.py即可测试，第24行为示例的query，调用do_query可以进行示例查询

## 数据格式说明

1. 数据格式为一个二维数组即可，每一行代表一个简单查询（各个属性只有上界，没有下届）以及其对应的值。假设有N个属性，那么一行数据就有N+1个值，第一个值为正则后的查询人次结果，而剩下的值就一一对应该简单查询的每个属性的上界。
2. 正则话。对于查询数量（第一个值）的正则化，应该是该简单查询获得人次除以库中所有有效人次。对于连续性的属性，就直接正则化到[0,1]；对于离散型的属性，预先给定好离散属性值的顺序，然后给予0,1,2…；之后再正则化即可。
3. 所以，所谓的网络，就是输入N个属性的正则化后的值，然后输出一个人次的比值。

## 代码说明（按照文件）

1. global.py: 定义了一些全局变量，以及偷懒直接在这里定义示例训练数据集
2. shuffle.py: 定义了如何将一维的训练数据通过不断重排列，然后将各种排列整合后获得二位数据。具体操作，比如属性数量为10个，那么我们就会选择10个不同的重排列（按照一定规则生成，且对于所有训练输入都一致），然后将他们vstack之后获得10*10的二维图像，这个二维图像才是真正的输入。这样shuffle的好处，是可以让原来一维相距较远的属性值通过重排列之后，在二维距离上相对较近，从而使得使用卷积神经网络更为合理
3. net.py: 网络的定义，全连接神经网络不利于泛化，所以这里以卷积神经网络为主，大致就是 卷积，池化，卷积，全连接，全连接。具体网络配置可以参见net.py的模型定义。相对使用了比较简单的网络，避免参数过多过拟合。
4. train.py: 就是pytorch的基本训练框架，并没有什么太多的花样
5. compose.py: 这个文件是用来文件复杂查询到简单查询，然后并且组合简单查询的结果获得复杂查询的结果。所谓的复杂查询就是每个属性既有上界，也有下界。因为一个包含n个属性的复杂query可以转化为2^n个简单查询，可以利用容斥定律来考虑这个问题（下一节细谈）
6. query.py: 利用compose.py和train.py获得model进行具体query的查询示意

## 复杂查询的转化

我们使用三个属性作为举例（代码注释中使用了2个属性举例）

同时，使用[0, A1 ] ^ [0, A2] ^ [0, A3] 这样的形式表示简单查询，其中A1,A2,A3就是对应属性的上界。显然简单查询是可以通过直接扔给神经网络获取对应的值的。

另外，使用R= [S1, E1] ^ [S2, E2] ^ [S3, E3] 这样的形式表示复杂查询，其中Si为下界，Ei为上界

我们可以把上述的查询看做是三维空间的一个范围，所以我们现在的目标就是把复杂查询对应的范围分解为不同的简单查询的范围的组合。

首先显然可以分解为以下8 = 2^3个组合，因为每个简单查询的每个属性分别有上界和下界的2个选择，

0. 0b000 => R0 = [0, E1] ^ [0, E2] ^ [0, E3]
1. 0b001 => R1 = [0, E1] ^ [0, E2] ^ [0, S3]
2. 0b010 => R2 = [0, E1] ^ [0, S2] ^ [0, E3]
3. 0b011 => R3 = [0, E1] ^ [0, S2] ^ [0, S3]
4. 0b100 => R4 = [0, S1] ^ [0, E2] ^ [0, E3]
5. 0b101 => R5 = [0, S1] ^ [0, E2] ^ [0, S3]
6. 0b110 => R6 = [0, S1] ^ [0, S2] ^ [0, E3]
7. 0b111 => R7 = [0, S1] ^ [0, S2] ^ [0, S3]

由容斥原理， 我们可以知道 R = R0 - R1 - R2 - R3 + R4 + R5 + R6 - R7

通过以上例子，我们来分析以下每一个简单查询如何生成以及最后如何将对一个的查询结果组合成复杂查询

首先就是分解操作，我们通过遍历0-> 2^n-1  这 2^n 个数字，然后根据其二进制表现形式，如上述的5，二进制就是0b101，然后根据二进制表示的第几位就代表第集个属性应该取上界还是下界作为简单查询的上界，比如0b101，第一个属性对应的位置是1，那么就去下界，然后第二个属性是0，就取上界，从而构成了R5。

最后对于上述2^n个简单查询，我们都获得了其输出，然后问题就是如何组合呢？

这个也非常简单，其实就是对他们进行加或者减，而其中每一项的符号取决与他们的二进制表示中含有1的个数，然后1的个数是偶数，那么就是+，否则就是-。所以在上述例子中， R5的二进制表示是0b101，其中包含偶数个1，所以在上述复杂查询R的组合中R5是取+。其他同理



# 以上就是所有的内容

 

