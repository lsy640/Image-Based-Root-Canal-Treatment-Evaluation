# train.py

import torch
import torch.nn as nn
from PyQt5.QtCore import QThread, pyqtSignal
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet34, resnet101, resnet50
import torchvision.models.resnet
# 定义一个线程类

class Resnet_Thread(QThread):
    # 自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self, in_dir, out_dir, epochs, lr, batch_size, parent=None):
        super(Resnet_Thread, self).__init__(parent)
        self.train_path = in_dir
        self.out_path = out_dir
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    # run函数是子线程中的操作，线程启动后开始执行
    # def run(self):
    # 发射自定义信号
    # 通过emit函数将参数i传递给主线程，触发自定义信号
    # self.finishSignal.emit(str(i))  # 注意这里与_signal = pyqtSignal(str)中的类型相同
    def run(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.2369732, 0.2066169, 0.043554377], [0.24172652, 0.37390167, 0.16018319]),
            ]),  # 参数
            "val": transforms.Compose([
                transforms.Resize(224),  # 将最小边长缩放到192
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.2369732, 0.2066169, 0.043554377], [0.24172652, 0.37390167, 0.16018319])
            ])}

        data_root = os.getcwd()
        image_path = self.train_path  # root data set path

        train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                             transform=data_transform["train"])
        train_num = len(train_dataset)


        root_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in root_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        batch_size = self.batch_size
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=0)

        validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=0)

        net = resnet50(num_classes=4)

        # load pretrain weights

        model_weight_path = "./models/resNet50.pth"
        missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)#载入模型参数

        for param in net.parameters():
            param.requires_grad = False
        # change fc layer structure

        inchannel = net.fc.in_features
        net.fc = nn.Linear(inchannel, 4)


        net.to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=self.lr)


        best_acc = 0.0
        save_path = self.out_path+'/resNet50_model.pth'
        # save_path = './resNet101.pth'

        for epoch in range(self.epochs):
            print('Epoch [%d/%d]' % (epoch, self.epochs))
            # train
            net.train()
            running_loss = 0.0
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # print train process
                rate = (step + 1) / len(train_loader)
                a = "*" * int(rate * 40)
                b = "." * int((1 - rate) * 40)
                print("train loss:%.5f " % loss)
                print("{:^3.0f}%[{}->{}]".format(int(rate * 100), a, b), end="")
            print()

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                for val_data in validate_loader:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))  # eval model only have last output layer
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += (predict_y == val_labels.to(device)).sum().item()
                val_accurate = acc / val_num
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(net.state_dict(), save_path)
                print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, running_loss / step, val_accurate))

            self.finishSignal.emit(epoch + 1)

        print('Finished Training')
