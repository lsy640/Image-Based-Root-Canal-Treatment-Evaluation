
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from qtpy import QtCore
from page_maincode import MainWindowActions

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 修复中文路径下Qt平台插件找不到的问题
import PyQt5
_qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_plugin_path


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # 界面的入口，在这里需要定义QApplication对象
    app = QApplication(sys.argv)
    # 显示创建的界面
    MainWindow = MainWindowActions()  # 创建窗体对象
    MainWindow.show()  # 显示窗体

    sys.exit(app.exec_())  # 程序关闭时退出进程

