# 基于PointRCNN修改的3D车辆检测

## PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
(原项目 模型训练 论文相关)->[传输门](https://github.com/sshaoshuai/PointRCNN)

# 魔改的目的就是为了能使用纯雷达检测,不再需要相机和校准文件

# 同时还魔改一个标注工具,也是不需要相机和校准文件
  能生成KITTI格式的标签文件 [labelCloud](https://github.com/zhousheng-zser/labelCloud)

## 部署环境Linux:
### Red Hat/CentOS 7​​
* Linux (localhost 3.10.0-1160.71.1.el7.  2022 x86_64 GNU/Linux)
* Python Python 3.7.12
* PyTorch 1.0
* CUDA version cuda-10.2(版本越高问题越多) + gcc8.3.0(第三方库pointnet2,iou3d,roipool3d依赖) + gcc-11.4.0(测试训练程序依赖)  ) 
* 别的python环境参考[博客1](https://blog.csdn.net/lixushi/article/details/118728278)和[博客2](https://blog.csdn.net/u014173215/article/details/123856191)

### Install 
* 参考原作者

## 部署环境nvidia盒子JetPack 5.1.3:
* Linux (ubuntu 5.15.136-tegra SMP PREEMPT PDT 2024 aarch64 GNU/Linux)
* docker (Ubuntu22.04.4 LTS (Jammy Jellyfish))
* Python 3.8.20
* PyTorch 1.0
* CUDA version cuda_11.8.r11.8
* [requirements.txt](https://github.com/zhousheng-zser/point_cloud_detect/blob/master/requirements.txt)

### Install 
* 参考原作者

## 原作者的预训练模型
下载地址 [here(~15MB)](https://drive.google.com/file/d/1aapMXBkSn5c5hNTDdRNI74Ptxfny7PuC/view?usp=sharing),

该模型是在训练集(3712 个样本)上进行训练，并在验证集(3769 个样本)和测试集(7518 个样本)上进行评估。在验证集上的表现情况如下:
```
Car AP@0.70, 0.70, 0.70:
bbox AP:96.91, 89.53, 88.74
bev  AP:90.21, 87.89, 85.51
3d   AP:89.19, 78.85, 77.91
aos  AP:96.90, 89.41, 88.54
```
## 我的训练模型
下载地址 [here(~22MB)](https://github.com/zhousheng-zser/point_cloud_detect/blob/master/tools/PointRCNN.pth)

将原训练模型中的各种车的分类改为了'Car',  又新加入2000+的大车和小车样本.主要是提高货车,卡车的检出率.

## 快速测试:
```
cd tools
python receiver-offline.py
# 可视化结果在../detect_clouds/
```