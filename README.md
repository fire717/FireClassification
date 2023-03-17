# Fire: Deep Learning for lazy humans

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE)

## 一、前言
Fire is a deep learning Framework written in Python and used for Image Classification task, running on top of the machine learning platform Pytorch.

Read the source code as documentation.

## 二、使用

### 2.1 训练
1. 下载[fashion mnist](https://github.com/zalandoresearch/fashion-mnist)数据集的四个压缩包放到data目录下，运行`python scripts/make_fashionmnist.py`自动提取图片并划分类别、验证集
2. 执行python train.py 训练
3. 执行python evaluate.py 测试（在config设置训练好的模型路径）

### 2.2 优化
* 迁移学习，下载对应模型的预训练模型，把路径填入config.py中
* 调整不同的模型、尺寸、优化器等等

### 2.3 自定义网络结构
依次修改fire/model.py相应代码即可。

## 三、功能
### 3.1.数据加载
* 文件夹形式
* csv标签形式
* 其它自定义形式需手动修改代码

### 3.2.支持网络

* Resnet系列，Densenet系列，VGGnet系列等所有[pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)支持的网络
* [Mobilenetv2](https://pytorch.org/docs/stable/torchvision/models.html?highlight=mobilenet#torchvision.models.mobilenet_v2)，[Mbilenetv3](https://github.com/kuan-wang/pytorch-mobilenet-v3)，ShuffleNetV2，[MicroNet](https://github.com/liyunsheng13/micronet)
* [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
* [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
* [TIMM](https://github.com/huggingface/pytorch-image-models)



### 3.3.优化器
* Adam  
* SGD 
* AdaBelief 
* [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
* AdamW

### 3.4.学习率衰减
* ReduceLROnPlateau
* StepLR
* MultiStepLR
* SGDR

### 3.5.损失函数
* 交叉熵
* Focalloss

### 3.6.其他
* Metric(acc, F1)
* 训练日志保存
* 交叉验证
* 梯度裁剪
* earlystop
* weightdecay
* 按文件夹设置分类标签、读取csv标签
* 冻结/解冻 除最后的全连接层的特征层
* labelsmooth

 

## 四、Update
* 2022.7 [v1.0] (根据这半年打比赛经验，增加一些东西，删除一些几乎不用的东西。) 增加convnext、swin transformer、半精度训练，删除mobileformer，删除日志、tensorboard（习惯用文档记录），优化readme
* 2021.8 [v0.9] 增加micronet和测试结果，增加rk3399测速
* 2021.8 [v0.8] 增加mobileformer，加入fashion mnist数据集使用demo，方便测试各种模型，同时加入部分网络的训练结果

## 五、To Do
* 完善Readme
* 增加使用文档
* 彻底分离用户自定义部分的代码
* 特征图热力图可视化(grad-cam)
* Dynamtic ReLU 

## 六、参考资源
1. [albumentations](https://github.com/albumentations-team/albumentations)
2. [warmup](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
3. [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
