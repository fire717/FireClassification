# Fire: Deep Learning for lazy humans

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE)

## 一、前言
Fire is a deep learning API written in Python, running on top of the machine learning platform Pytorch.

Read the documentation at source code.

## 二、功能
### 1.已支持网络
#### 分类
* Resnet系列，Densenet系列，VGGnet系列等所有[pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)支持的网络
* [mobilenetv2](https://pytorch.org/docs/stable/torchvision/models.html?highlight=mobilenet#torchvision.models.mobilenet_v2)，[mobilenetv3](https://github.com/kuan-wang/pytorch-mobilenet-v3)
* [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)

### 2.优化器
* Adam  
* SGD 
* AdaBelief 
* Ranger

### 3.学习率衰减
* ReduceLROnPlateau
* StepLR
* SGDR

### 4.损失函数
* 交叉熵
* Focalloss

### 5.其他
* Metric(acc, F1)
* 训练日志保存
* 交叉验证
* 梯度裁剪
* earlystop
* weightdecay
* 按文件夹设置分类标签、读取csv标签
* 冻结/解冻 除最后的全连接层的特征层
* tensorboard可视化
* labelsmooth

### 

## 三、To Do
* 完善Readme
* 增加使用文档
* 彻底分离用户自定义部分的代码
* tensorboard可视化
* 特征图热力图可视化

## 四、参考资源
1. [warmup](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
2. [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)