# Fire: Deep Learning for lazy humans

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE)

## 一、前言
Fire is a deep learning API written in Python, running on top of the machine learning platform Pytorch.

Read the documentation at source code.

## 二、使用

### 2.1 训练
1. 下载[fashion mnist](https://github.com/zalandoresearch/fashion-mnist)放到data目录下，运行make_fashionmnist.py自动提取图片并划分类别、验证集
2. 拷贝 fire/examples/config.py.example到根目录并去掉后缀.example，然后根据需要修改相应参数、配置
3. 拷贝 fire/examples/dataaug_user.py.example到fire目录并去掉后缀.example，然后根据需要修改相应数据增强（默认无增强）
4. 执行python train.py 训练
5. 执行python evaluate.py 测试（在config设置模型路径）

### 2.2 自定义网络结构
依次修改fire/model.py相应代码即可。

## 三、功能
### 1.已支持网络
#### 分类
* Resnet系列，Densenet系列，VGGnet系列等所有[pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)支持的网络
* [Mobilenetv2](https://pytorch.org/docs/stable/torchvision/models.html?highlight=mobilenet#torchvision.models.mobilenet_v2)，[Mbilenetv3](https://github.com/kuan-wang/pytorch-mobilenet-v3)，[Mobileformer](https://github.com/slwang9353/MobileFormer)，[MicroNet](https://github.com/liyunsheng13/micronet)
* [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)

#### 性能测试
网络|最佳epoch|模型大小(MB)|acc(%)|rk3399(ms)
---|---|---|---|---
Mobilenetv2-1.0|10|13.7|94.3|90
Mobilenetv3-small-1.0|15|11.4|92.9|95
MobileFormer-96|9|4.82|21.9|-
MicroNet-M0|14|1.15|90.6|20
MicroNet-M1|28|2.38|92.6|26
MicroNet-M2|11|3.45|92.6|45
MicroNet-M3|14|4.18|94.0|65

* 测试说明：数据集为[fashion mnist](https://github.com/zalandoresearch/fashion-mnist)，本身为28x28，为了结合实际缩放到224x224，训练集训练，测试集测试。其中训练集后1000张划分为验证集，使用earlystop（patience=7）选择最佳模型去测试。统一使用SGD，lr0.001，bs=64，epoch100，ReduceLROnPlateau（3，0.1），交叉熵；都不使用预训练，都不使用数据增强
* MobileFormer难道是打开方式不对？测试了下原repo训练mnist，报错，且无法转onnx

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

 

## 四、Update
* 2021.8 v0.0.9 增加micronet和测试结果，增加rk3399测速

* 2021.8 v0.0.8 增加mobileformer，加入fashion mnist数据集使用demo，方便测试各种模型，同时加入部分网络的训练结果

## 五、To Do
* 完善Readme
* 增加使用文档
* 彻底分离用户自定义部分的代码
* tensorboard可视化
* 特征图热力图可视化(grad-cam)
* Dynamtic ReLU 

## 六、参考资源
1. [albumentations](https://github.com/albumentations-team/albumentations)
2. [warmup](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
3. [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
4. [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
