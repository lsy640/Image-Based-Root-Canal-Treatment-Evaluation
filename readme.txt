该文件夹为根管治疗评价系统可视化界面代码。
程序使用python编写，神经网络采用pytorch库，可视化界面采用PyQt5。
文件夹：
Inputs文件夹为分割图片。root_dataset 为根管数据。images存放长宽比为1:1加黑边预处理输入图像，mask存储掩膜图形。0：牙齿区域掩膜；1：根管区域掩膜。
Models文件夹存放训练好的模型权重文件。
Outputs存放分割图。
Rgb存放叠加法特征融合图
代码文件：
Main.py: 主函数，运行该函数启动。
archs.py: Unet++模型结构代码
model.py: ResNet模型结构代码
dataset.py：数据集处理函数
losses.py: 损失函数
metrics.py: 计算IoU函数
utils.py: 计算平均和当前值
train_c.py：分类网络训练函数
val.py:  分割函数
train_u.py：分割网络训练函数
Predict.py: 分类函数
page_main.py：主界面代码
page_maincode.py：主界面控制代码
page1.py：评价界面代码
page1_code.py：评价界面控制代码
page2.py：训练界面代码
page2_code.py：训练界面控制代码
trans_3gto1.py：叠加法特征融合函数