import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from page1 import *
import page_maincode


class ValWindowActions(QMainWindow, Ui_ValWindow):
    def __init__(self, parent=None):
        super(ValWindowActions, self).__init__(parent)
        self.setupUi(self)
        self.re.clicked.connect(self.click_re_button)

    def click_re_button(self):
        """点击返回按钮，跳转到主界面"""
        # 实例化第二个界面的后端类，并对第二个界面进行显示
        self.main_window = page_maincode.MainWindowActions()
        # 显示主界面
        self.main_window.show()
        # 关闭评价界面
        self.close()