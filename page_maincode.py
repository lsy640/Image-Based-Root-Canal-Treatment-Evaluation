import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import page_main
import page1_code
import page2_code


class MainWindowActions(QMainWindow, page_main.Ui_mianpage):
    def __init__(self, parent=None):
        super(MainWindowActions, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_val.clicked.connect(self.click_val_button)
        self.pushButton_train.clicked.connect(self.click_train_button)

    def click_val_button(self):
        """点击评价按钮，跳转到相应界面"""
        # 实例化第二个界面的后端类，并对第二个界面进行显示
        self.val_window = page1_code.ValWindowActions()
        # 显示评价界面
        self.val_window.show()
        # 关闭主界面
        self.close()

    def click_train_button(self):
        """点击训练按钮，跳转到相应界面"""
        # 实例化第二个界面的后端类，并对第二个界面进行显示
        self.tra_window = page2_code.TraWindowActions()
        # 显示训练界面
        self.tra_window.show()
        # 关闭主界面
        self.close()

    '''def closeEvent(self, event):        #关闭窗口触发以下事件
        a = QMessageBox.question(self, '退出', '你确定要退出吗?', QMessageBox.No | QMessageBox.Yes, QMessageBox.Yes)      #
        # "退出"代表的是弹出框的标题,"你确认退出.."表示弹出框的内容
        if a == QMessageBox.Yes:
            event.accept()        #接受关闭事件
        else:
            event.ignore()        #忽略关闭事件'''


