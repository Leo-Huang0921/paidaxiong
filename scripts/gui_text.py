import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt


# 所有的自定义界面，都要继承自 QWidget（基础窗口）或 QMainWindow（带菜单栏的主窗口）
class SimpleWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()  # 初始化界面

    def init_ui(self):
        # 1. 设置窗口的基本属性
        self.setWindowTitle('我的第一个 GUI 程序')
        self.resize(300, 200)  # 宽 300，高 200

        # 2. 创建垂直布局器 (Vertical Box Layout)
        # 布局器的作用是自动排列控件，不用你手动算像素位置
        layout = QVBoxLayout()

        # 3. 创建控件（积木）
        self.title_label = QLabel('欢迎来到 GUI 的世界！', self)
        self.title_label.setAlignment(Qt.AlignCenter)  # 文字居中

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("请输入你的名字...")  # 占位提示词

        self.btn_submit = QPushButton('点击问好', self)

        # 4. 把控件像塞俄罗斯方块一样，按顺序塞进布局器里
        layout.addWidget(self.title_label)
        layout.addWidget(self.input_box)
        layout.addWidget(self.btn_submit)

        # 5. 把布局器应用到当前窗口上
        self.setLayout(layout)

        # 6. 绑定事件（重点：信号与槽）
        # 当按钮被点击(clicked)时，连接(connect)到自定义的函数(show_greeting)
        self.btn_submit.clicked.connect(self.show_greeting)

    # 定义按钮按下的动作
    def show_greeting(self):
        # 获取输入框里的文字
        user_name = self.input_box.text()

        # 简单的逻辑判断
        if user_name.strip() == "":
            # 弹出警告框
            QMessageBox.warning(self, "警告", "名字不能为空哦！")
        else:
            # 弹出信息框
            QMessageBox.information(self, "打招呼", f"你好，{user_name}！很高兴认识你。")


# ================= 程序的绝对入口 =================
if __name__ == '__main__':
    # 1. 创建应用程序对象（每个 PyQt5 程序必须有且只有一个）
    app = QApplication(sys.argv)

    # 2. 实例化我们写好的窗口类
    window = SimpleWindow()

    # 3. 显示窗口（默认是隐藏的）
    window.show()

    # 4. 进入程序的主循环（让窗口一直亮着，等待用户操作，直到点击右上角的 X）
    sys.exit(app.exec_())