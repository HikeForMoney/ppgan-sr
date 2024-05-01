HIKE | [原始README](./README_en.md)
# 一、背景
使用 ppgan 的 esrgan 做图像超分辨率，参考[esrgan教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/single_image_super_resolution.md)。

# 二、准备
## 2.1 环境
```
paddle
ppgan
```

## 2.2 数据准备
### 2.2.1 数据下载
|数据集|下载链接|
|----|----|
|DIV2K|阿里云盘 DIV2K.tar|
|Set5|阿里云盘 Set5.zip|
|Set14|阿里云盘 Set4.zip|

### 2.2.2 目录格式
```
  PaddleGAN
    ├── data
        ├── DIV2K
              ├── DIV2K_train_HR
              ├── DIV2K_train_LR_bicubic
              |    ├──X2
              |    ├──X3
              |    └──X4
              ├── DIV2K_valid_HR
              ├── DIV2K_valid_LR_bicubic
            Set5
              ├── GTmod12
              ├── LRbicx2
              ├── LRbicx3
              ├── LRbicx4
              └── original
            Set14
              ├── GTmod12
              ├── LRbicx2
              ├── LRbicx3
              ├── LRbicx4
              └── original
            ...
```
### 2.2.3 数据目录预处理
```
  python data/process_div2k_data.py --data-root data/DIV2K
```

# 三、运行
## 3.1 训练
```
bash train.sh
```

## 3.2 推理
```
bash infra.sh
```