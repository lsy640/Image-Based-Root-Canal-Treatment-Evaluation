import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import yaml

# https://github.com/albumentations-team/albumentations
# pip install -U albumentations
# python3.6+
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
import page2
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

from PyQt5.QtCore import QThread, pyqtSignal
import page2_code

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

"""

指定参数：
--dataset dsb2018_96 
--arch NestedUNet

"""

# 定义一个线程类
class Unet_Thread(QThread):
    # 自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)

    # 带一个参数t
    def __init__(self, in_dir, out_dir, epochs, lr, batch_size, parent=None):
        super(Unet_Thread, self).__init__(parent)
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

    # 定义训练参数
    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--name', default=None,
                            help='model name: (default: arch)')
        parser.add_argument('--epochs', default=200, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('-b', '--batch_size', default=8, type=int,
                            metavar='N', help='mini-batch size (default: 16)')

        # model
        parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                            choices=ARCH_NAMES,
                            help='model architecture: ' +
                                 ' | '.join(ARCH_NAMES) +
                                 ' (default: NestedUNet)')
        parser.add_argument('--deep_supervision', default=False, type=str2bool)
        parser.add_argument('--input_channels', default=3, type=int,
                            help='input channels')
        parser.add_argument('--num_classes', default=2, type=int,
                            help='number of classes')
        parser.add_argument('--input_w', default=192, type=int,
                            help='image width')
        parser.add_argument('--input_h', default=192, type=int,
                            help='image height')

        # loss
        parser.add_argument('--loss', default='BCEDiceLoss',
                            choices=LOSS_NAMES,
                            help='loss: ' +
                                 ' | '.join(LOSS_NAMES) +
                                 ' (default: BCEDiceLoss)')

        # dataset
        parser.add_argument('--dataset', default='root_dataset',
                            help='dataset name')
        parser.add_argument('--img_ext', default='.jpg',
                            help='image file extension')
        parser.add_argument('--mask_ext', default='.png',
                            help='mask file extension')

        # optimizer
        parser.add_argument('--optimizer', default='SGD',
                            choices=['Adam', 'SGD'],
                            help='loss: ' +
                                 ' | '.join(['Adam', 'SGD']) +
                                 ' (default: Adam)')
        parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='momentum')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weight decay')
        parser.add_argument('--nesterov', default=False, type=str2bool,
                            help='nesterov')

        # scheduler
        parser.add_argument('--scheduler', default='CosineAnnealingLR',
                            choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
        parser.add_argument('--min_lr', default=1e-5, type=float,
                            help='minimum learning rate')
        parser.add_argument('--factor', default=0.1, type=float)
        parser.add_argument('--patience', default=2, type=int)
        parser.add_argument('--milestones', default='1,2', type=str)
        parser.add_argument('--gamma', default=2 / 3, type=float)
        parser.add_argument('--early_stopping', default=-1, type=int,
                            metavar='N', help='early stopping (default: -1)')

        parser.add_argument('--num_workers', default=0, type=int)

        config = parser.parse_args()

        return config

    def train(self, config, train_loader, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter()}

        model.train()
        step = 0
        for input, target, _ in train_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)  # 计算损失
                iou = iou_score(output, target)  # 计算iou

            # 计算梯度和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))  # 计算平均损失
            avg_meters['iou'].update(iou, input.size(0))  # 计算平均iou

            # print train process
            rate = (step + 1) / len(train_loader)
            step += 1
            a = "*" * int(rate * 40)
            b = "." * int((1 - rate) * 40)
            print("train loss:%.5f" % loss + "  iou:%.5f" % iou)
            print("{:^3.0f}%[{}->{}]".format(int(rate * 100), a, b), end="")
        print()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg)])

    def validate(self, config, val_loader, model, criterion):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter()}

        # switch to evaluate mode
        model.eval()
        step = 0
        with torch.no_grad():
            for input, target, _ in val_loader:
                input = input.cuda()
                target = target.cuda()

                # compute output
                if config['deep_supervision']:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou = iou_score(outputs[-1], target)
                else:
                    output = model(input)
                    loss = criterion(output, target)
                    iou = iou_score(output, target)

                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))

                # print train process
                rate = (step + 1) / len(val_loader)
                step += 1
                a = "*" * int(rate * 40)
                b = "." * int((1 - rate) * 40)
                print("val loss:%.5f " % loss + "  iou:%.5f" % iou)
                print("{:^3.0f}%[{}->{}]".format(int(rate * 100), a, b), end="")
            print()


        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg)])

    def run(self):
        config = vars(self.parse_args())

        if config['name'] is None:
            if config['deep_supervision']:
                config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
            else:
                config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
        os.makedirs('models/%s' % config['name'], exist_ok=True)

        print('-' * 20)
        for key in config:
            print('%s: %s' % (key, config[key]))
        print('-' * 20)

        with open('models/%s/config.yml' % config['name'], 'w') as f:
            yaml.dump(config, f)

        # define loss function (criterion)
        if config['loss'] == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss().cuda()  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
        else:
            criterion = losses.__dict__[config['loss']]().cuda()

        cudnn.benchmark = True

        # create model
        print("=> creating model %s" % config['arch'])
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision'])

        model = model.cuda()

        params = filter(lambda p: p.requires_grad, model.parameters())
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                params, lr=self.lr, weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(params, lr=self.lr, momentum=config['momentum'],
                                  nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        else:
            raise NotImplementedError

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=config['min_lr'])
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                       verbose=1, min_lr=config['min_lr'])
        elif config['scheduler'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[int(e) for e in config['milestones'].split(',')],
                                                 gamma=config['gamma'])
        elif config['scheduler'] == 'ConstantLR':
            scheduler = None
        else:
            raise NotImplementedError

        # 数据加载
        img_ids = glob(os.path.join(self.train_path, 'images', '*' + config['img_ext']))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)  # 将样本分为测试集和训练集，测试集占比0.2
        # 数据增强：
        train_transform = Compose([
            A.RandomRotate90(),
            A.Flip(),
            OneOf([
                transforms.HueSaturationValue(),
                transforms.RandomBrightnessContrast(),
                # transforms.RandomBrightnessContrast(),
            ], p=1),  # 按照归一化的概率选择执行哪一个
            A.Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ], is_check_shapes=False)

        val_transform = Compose([
            A.Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ], is_check_shapes=False)
        # 准备输入数据集和标记数据集
        train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=os.path.join(self.train_path, 'images'),
            mask_dir=os.path.join(self.train_path, 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=train_transform)
        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=os.path.join(self.train_path, 'images'),
            mask_dir=os.path.join(self.train_path, 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True)  # 不能整除的batch是否就不要了
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False)

        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_iou', []),
        ])

        best_iou = 0
        trigger = 0
        for epoch in range(self.epochs):

            print('Epoch [%d/%d]' % (epoch, self.epochs))

            # 一个周期训练
            train_log = self.train(config, train_loader, model, criterion, optimizer)
            # 验证集上评估
            val_log = self.validate(config, val_loader, model, criterion)

            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler.step()
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])

            print('loss %.4f - iou %.4f \r val_loss %.4f - val_iou %.4f'
                  % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

            log['epoch'].append(epoch)
            log['lr'].append(self.lr)
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])

            pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                     config['name'], index=False)

            trigger += 1

            if val_log['iou'] > best_iou:
                torch.save(model.state_dict(), self.out_path + '/Unet++model.pth')
                print(self.out_path)
                best_iou = val_log['iou']
                print("=> saved best model")
                trigger = 0

            # 早停止（如果训练结果不更新的周期都大于参数early_stopping，则停止。防止模型退化）
            if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                print("=> early stopping")
                break

            self.finishSignal.emit(epoch + 1)
            torch.cuda.empty_cache()


