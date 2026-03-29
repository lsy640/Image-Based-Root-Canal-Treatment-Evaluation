**中文 | [English](README.md)**

# 基于图像的根管治疗效果评价系统

本项目是一个基于深度学习的根管治疗效果自动评价系统。系统通过对根尖 X 光片进行图像分割与分类，自动评估根管填充质量，输出 A/B/C/D 四个等级的评价结果。采用 PyQt5 构建可视化操作界面，支持一键评价和模型训练功能。

## 系统架构

系统采用 **"分割 + 特征融合 + 分类"** 的三阶段流水线架构：

```
根尖 X 光片输入
       │
       ▼
  ┌──────────┐
  │  UNet++   │  ── 语义分割：提取牙齿区域掩膜 + 根管区域掩膜
  └──────────┘
       │
       ▼
  ┌──────────┐
  │ 特征融合  │  ── 叠加法：将原图(R)、牙齿掩膜(G)、根管掩膜(B) 合成为 RGB 三通道图像
  └──────────┘
       │
       ▼
  ┌──────────┐
  │ ResNet50  │  ── 图像分类：对融合图像进行根管填充质量分级
  └──────────┘
       │
       ▼
  评价结果 (A/B/C/D)
```

### GUI 界面

系统提供三个操作界面：

- **主界面**：功能导航入口，可跳转至评价界面或训练界面
- **评价界面**：上传根尖片，选择模型，一键完成分割、融合、分类全流程，显示评价结果
- **训练界面**：选择数据集路径和输出路径，配置训练参数（epoch、学习率、batch size），支持 UNet++ 和 ResNet 模型训练

## 项目结构

```
├── main.py                 # 程序入口
├── requirements.txt        # 依赖包列表
├── class_indices.json      # 分类标签映射 {"0":"A", "1":"B", "2":"C", "3":"D"}
│
├── archs.py                # UNet++ (NestedUNet) 网络架构定义
├── model.py                # ResNet 网络架构定义 (支持 ResNet18/34/50/101/152)
├── dataset.py              # 自定义 Dataset，加载图像与掩膜
├── losses.py               # BCEDiceLoss 损失函数
├── metrics.py              # IoU、Dice 系数计算
├── utils.py                # 工具函数（AverageMeter 等）
│
├── train_u.py              # UNet++ 分割模型训练（QThread 子线程）
├── train_c.py              # ResNet 分类模型训练（QThread 子线程）
├── val.py                  # 分割模型推理/验证
├── predict.py              # 分类模型推理
├── trans_3gto1.py          # 叠加法特征融合（原图 + 牙齿掩膜 + 根管掩膜 → RGB）
│
├── page_main.py            # 主界面 UI 布局
├── page_maincode.py        # 主界面逻辑控制
├── page1.py                # 评价界面 UI 布局
├── page1_code.py           # 评价界面逻辑控制
├── page2.py                # 训练界面 UI 布局
├── page2_code.py           # 训练界面逻辑控制
│
├── inputs/                 # 分割数据集目录
│   └── root_dataset/
│       ├── images/         # 输入图像（1:1 长宽比，加黑边预处理）
│       └── masks/
│           ├── 0/          # 牙齿区域掩膜
│           └── 1/          # 根管区域掩膜
│
├── models/                 # 训练好的模型权重
│   ├── resNet50.pth        # ResNet50 分类模型权重
│   └── root_dataset_NestedUNet_woDS/
│       ├── model.pth       # UNet++ 分割模型权重
│       ├── config.yml      # 训练配置
│       └── log.csv         # 训练日志
│
├── outputs/                # 分割输出结果
│   └── root_dataset_NestedUNet_woDS/
│       ├── 0/              # 牙齿区域分割结果
│       └── 1/              # 根管区域分割结果
│
└── rgb/                    # 叠加法特征融合输出图像
```

## 环境要求与安装

### 环境要求

- **Python 3.9**（必须，PyTorch 在更高版本上可能存在兼容性问题）
- CUDA（可选，GPU 加速训练）

### 安装步骤

1. **创建虚拟环境**

```bash
python -m venv venv
```

2. **激活虚拟环境**

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

> **注意**：`albumentations` 必须使用 2.0 以下版本（`albumentations<2.0`），2.0 及以上版本存在 API 不兼容问题（`albumentations.augmentations.transforms` 模块被移除）。

### 已知问题

- **中文路径下 Qt 插件找不到**：如果项目路径包含中文字符，可能出现 `no Qt platform plugin could be initialized` 错误。`main.py` 中已添加修复代码，自动设置 `QT_QPA_PLATFORM_PLUGIN_PATH` 环境变量。
- **albumentations 形状检查**：图像和掩膜尺寸不一致时，`Compose` 可能报 `ValueError`。代码中已通过 `is_check_shapes=False` 参数解决。
- **GUI 界面未做自动缩放**：当前界面在 Windows 系统下以 100% 缩放创建，如遇字体不显示或显示不全的问题，请调整系统显示缩放比例。macOS 系统下也可能存在字体显示不全的问题。

## 使用说明

### 启动程序

```bash
python main.py
```

### 评价操作流程

1. 在主界面点击进入 **评价界面**
2. 上传待评价的根尖 X 光片
3. 选择分割模型（UNet++）和分类模型（ResNet50）的权重文件
4. 点击开始评价，系统将依次完成：
   - UNet++ 分割 → 提取牙齿区域和根管区域掩膜
   - 叠加法特征融合 → 生成 RGB 三通道融合图像
   - ResNet50 分类 → 输出填充质量等级
5. 查看评价结果（等级 + 置信度 + Top-4 预测概率）

### 训练操作流程

1. 在主界面点击进入 **训练界面**
2. 选择训练模型类型（UNet++ 分割 / ResNet 分类）
3. 设置数据集路径和模型输出路径
4. 配置训练参数（epochs、学习率、batch size）
5. 点击开始训练，界面显示训练进度

## 模型说明

### UNet++ 分割模型

- **网络结构**：NestedUNet（UNet++ 嵌套 U 型网络）
- **基础模块**：VGGBlock（Conv-BN-ReLU × 2）
- **通道配置**：[32, 64, 128, 256, 512]
- **输入尺寸**：192 × 192 × 3
- **输出类别**：2（牙齿区域 + 根管区域）
- **损失函数**：BCEDiceLoss（BCE + Dice 联合损失）
- **优化器**：SGD（lr=1e-3, momentum=0.9, weight_decay=1e-4）
- **学习率调度**：CosineAnnealingLR
- **数据增强**：随机旋转90度、翻转、色调/饱和度/亮度/对比度随机变换

### ResNet50 分类模型

- **网络结构**：ResNet50（50 层残差网络）
- **输入尺寸**：224 × 224 × 3（RGB 融合图像）
- **输出类别**：4（A / B / C / D）
- **图像归一化**：mean=[0.237, 0.207, 0.044], std=[0.242, 0.374, 0.160]

### 评价等级定义

| 等级 | 含义 |
|------|------|
| **A** | 根管填充恰填，密度均匀，填充效果良好 |
| **B** | 根管填充轻微欠填或超填，但在可接受范围内 |
| **C** | 根管填充明显不足或过度，需要关注 |
| **D** | 根管填充严重不合格，建议重新治疗 |

当 Top-1 预测置信度低于 0.7 时，系统输出模糊等级（如 A~B、B~C），表示结果介于两个等级之间。

### 特征融合方法 —— 叠加法

将分割结果与原图融合为单张 RGB 图像输入分类网络：

- **R 通道**：原始根尖片灰度图（192 × 192）
- **G 通道**：牙齿区域分割掩膜（192 × 192）
- **B 通道**：根管区域分割掩膜（192 × 192）

这种方法使分类网络能同时利用原始图像纹理信息和分割区域的空间位置信息。

## 实验结果

以下结果来自论文实验（数据集包含 500+ 张根尖 X 光片）：

### 分割性能

| 指标 | 数值 |
|------|------|
| 平均 IoU | 0.7634 |

### 分类性能（叠加法 + ResNet50）

| 指标 | 数值 |
|------|------|
| Top-1 准确率 | 59.40% |
| Top-2 准确率 | 86.22% |

## 参考与致谢

### 原始代码仓库

- **UNet++（pytorch-nested-unet）**：[https://github.com/4uiiurz1/pytorch-nested-unet](https://github.com/4uiiurz1/pytorch-nested-unet)

  本项目的 UNet++ 分割网络架构基于该仓库实现。

- **ResNet（PyTorch 官方 torchvision）**：[https://github.com/pytorch/vision](https://github.com/pytorch/vision)

  本项目的 ResNet 分类网络参考了 torchvision 中的 ResNet 实现。

### 参考论文

- Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). **UNet++: A Nested U-Net Architecture for Medical Image Segmentation.** *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, pp. 3-11.

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778.

### 相关技术

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI 框架
- [albumentations](https://github.com/albumentations-team/albumentations) - 图像增强库
- [OpenCV](https://opencv.org/) - 图像处理库
