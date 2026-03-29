import os
import sys

from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QTextCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QLabel, QMessageBox

from page2 import *
import page_maincode

from train_u import Unet_Thread
from train_c import Resnet_Thread


class Signal(QObject):
    text_update = pyqtSignal(str)

    def write(self, text):
        self.text_update.emit(str(text))
        # loop = QEventLoop()
        # QTimer.singleShot(100, loop.quit)
        # loop.exec_()


class TraWindowActions(QMainWindow, Ui_TrainWindow):
    global in_dir, out_dir
    in_dir = ""
    out_dir = ""
    global train_state
    train_state = 0

    def __init__(self, parent=None):
        super(TraWindowActions, self).__init__(parent)
        self.setupUi(self)
        self.re.clicked.connect(self.click_re_button)
        self.pushButton.clicked.connect(self.Input_File)
        self.pushButton_2.clicked.connect(self.Output_File)
        self.ok.clicked.connect(self.startAction)
        self.pushButton_3.clicked.connect(self.Stop)
        # 实时显示输出, 将控制台的输出重定向到界面中
        sys.stdout = Signal()
        sys.stdout.text_update.connect(self.updatetext)

    def updatetext(self, text):
        """
            更新textBrowser,将控制台输出到Qt界面
        """
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.textBrowser.append(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def click_re_button(self):
        """点击返回按钮，跳转到主界面"""
        if train_state == 0:  # 未开始训练
            # 实例化第二个界面的后端类，并对第二个界面进行显示
            self.main_window = page_maincode.MainWindowActions()
            # 显示主界面
            self.main_window.show()
            # 关闭训练界面
            self.close()
        else:
            msg_box = QMessageBox(QMessageBox.Information, '提示', '当前正在训练，若要返回请取消训练')
            msg_box.exec_()

    def Input_File(self):
        if train_state == 0:  # 未开始训练
            # 显示数据集格式窗口
            if self.comboBox.currentIndex() == 0:
                self.show_function("./ui-img/U_dataset.png")
            else:
                self.show_function("./ui-img/R_dataset.png")

            global in_dir
            in_dir = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择训练文件夹路径", "")
            self.lineEdit.setText(in_dir)

        else:
            msg_box = QMessageBox(QMessageBox.Information, '提示', '当前正在训练，若要选择请取消训练')
            msg_box.exec_()

    def Output_File(self):
        if train_state == 0:  # 未开始训练
            global out_dir
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择模型输出文件夹路径", "")
            self.lineEdit_2.setText(out_dir)
        else:
            msg_box = QMessageBox(QMessageBox.Information, '提示', '当前正在训练，若要选择请取消训练')
            msg_box.exec_()

    def show_function(self, jpgpath):
        dialog_fault = QDialog()
        dialog_fault.resize(700, 700)
        self.label = QtWidgets.QLabel(dialog_fault)
        self.label.setGeometry(QtCore.QRect(0, 0, 700, 600))
        self.label.setText("")
        self.label.setObjectName("label")
        # 给label添加背景图片
        jpg = QPixmap(jpgpath)
        self.label.setPixmap(jpg)
        # 图片自适应窗体大小
        self.label.setScaledContents(True)
        self.label.setText("")
        self.label.setObjectName("pic_root")
        self.pushButton = QtWidgets.QPushButton(dialog_fault)
        self.pushButton.setGeometry(QtCore.QRect(300, 630, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("确认")

        self.pushButton.clicked.connect(dialog_fault.close)
        dialog_fault.exec_()

    # 显示训练进度
    def dis_epoch(self, epo):
        global train_state
        self.progressBar.setValue(int(epo / epochs*100))
        #self.label_11.setText("%d%%" % (epo / epochs * 100))

        if epo == epochs:  # 训练结束
            self.comboBox.setEnabled(True)  # 可用相关组件
            self.spinBox.setEnabled(True)
            self.doubleSpinBox.setEnabled(True)
            self.spinBox_2.setEnabled(True)
            train_state = 0
            print("=>Finish training model")
            if self.comboBox.currentIndex() == 0:
                msg_box = QMessageBox(QMessageBox.Information, '提示', '训练完成，模型已保存到：%s/Unet++model.pth' % out_dir)
                msg_box.exec_()
            else:
                msg_box = QMessageBox(QMessageBox.Information, '提示', '训练完成，模型已保存到：%s/Resnet50_model.pth' % out_dir)
                msg_box.exec_()

    # 开始训练函数
    def startAction(self):
        global epochs, train_state
        if in_dir == "":
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '请选择训练数据集')
            msg_box.exec_()
            return
        elif out_dir == "":
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '请选择模型输出路径')
            msg_box.exec_()
            return
        elif train_state == 1:  # 正在训练
            msg_box = QMessageBox(QMessageBox.Information, '提示', '当前正在训练，若要重新训练请取消当前训练')
            msg_box.exec_()
            return

        train_state = 1  # 训练标志置1，表示正在训练
        self.comboBox.setEnabled(False)  # 禁用部分组件菜单
        self.spinBox.setEnabled(False)
        self.doubleSpinBox.setEnabled(False)
        self.spinBox_2.setEnabled(False)

        epochs = self.spinBox.value()
        lr = self.doubleSpinBox.value()
        batch_size = self.spinBox_2.value()

        self.label_7.setText("训练进度：")
        self.label_10.setText("训练详细输出：")
        #self.label_11.setText("0%")
        self.pushButton_3.setText("取消")
        self.progressBar.show()
        self.pushButton_3.show()
        self.textBrowser.show()
        self.progressBar.setValue(0)

        if self.comboBox.currentIndex() == 0:
            self.trainUnet_thread = Unet_Thread(in_dir, out_dir, epochs, lr, batch_size)  # 开始一个新的进程运行训练函数
            self.trainUnet_thread.finishSignal.connect(self.dis_epoch)  # 完成一个周期训练后，返回当前周期，启动显示函数
            self.trainUnet_thread.start()  # 进程开始
        else:
            self.trainResnet_thread = Resnet_Thread(in_dir, out_dir, epochs, lr, batch_size)  # 开始一个新的进程运行训练函数
            self.trainResnet_thread.finishSignal.connect(self.dis_epoch)  # 完成一个周期训练后，返回当前周期，启动显示函数
            self.trainResnet_thread.start()  # 进程开始

    # 终止训练
    def Stop(self):
        global train_state
        if train_state == 0:
            return

        a = QMessageBox.question(self, '退出', '你确定要取消训练吗?', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)  # "退出"代表的是弹出框的标题,"你确认退出.."表示弹出框的内容
        if a == QMessageBox.Yes:  # 接受关闭事件
            self.comboBox.setEnabled(True)  # 可用相关组件
            self.spinBox.setEnabled(True)
            self.doubleSpinBox.setEnabled(True)
            self.spinBox_2.setEnabled(True)
            train_state = 0
            if self.comboBox.currentIndex() == 0:
                self.trainUnet_thread.terminate()  # 终止Unet训练线程
                print("=>Stop training Unet++ model")
            else:
                self.trainResnet_thread.terminate()  # 终止Resnet训练线程
                print("=>Stop training Resnet50 model")
