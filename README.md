# demo04_ResNet
# ResNet网络详解

## 一、概述

ResNet在2015年由微软实验室提出，斩获当年ImageNet竞赛中分类任务第一名，目标检测第一名。获得COCO数据集中目标检测第一名，图像分割第一名。

![image-20220416111455720](/home/zs/.config/Typora/typora-user-images/image-20220416111455720.png)

## 二、亮点

- 超深的网络结构(突破1000层)

![image-20220416111628212](/home/zs/.config/Typora/typora-user-images/image-20220416111628212.png)

- 提出residual模块
![image-20220416111721656](/home/zs/.config/Typora/typora-user-images/image-20220416111721656.png)
- 使用Batch Normalization加速训练(丢弃dropout)

## 三、网络详解

### 1.residual结构

#### 实线shortcut结构

![image-20220416112050953](/home/zs/.config/Typora/typora-user-images/image-20220416112050953.png)

左边这幅图原来是以输入channel为64，3x3卷积层卷积个数也是64为例的，这里为了方便对比都改成256

通过参数对比可以知道，右边的结构更加节省参数

#### 虚线shortcut结构

![image-20220416112734183](/home/zs/.config/Typora/typora-user-images/image-20220416112734183.png)

![image-20220416113310712](/home/zs/.config/Typora/typora-user-images/image-20220416113310712.png)

## 四、网络结构

![image-20220416112910887](/home/zs/.config/Typora/typora-user-images/image-20220416112910887.png)

![image-20220416112923991](/home/zs/.config/Typora/typora-user-images/image-20220416112923991.png)	
