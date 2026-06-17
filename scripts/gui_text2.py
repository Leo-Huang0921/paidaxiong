import sys
# 1. 这里的 QWidget 换成 QDialog
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from PyQt5 import uic


# 2. 这里的继承也换成 QDialog
class MyWindow(QDialog):
    def __init__(self):
        super().__init__()

        # 加载图纸
        uic.loadUi('login.ui', self)

        # ================= 2. 给按钮绑功能 =================
        # .clicked.connect() 的意思是：当按钮被点击时，去执行括号里的函数
        # 注意：这里的 btn_login 必须和你在 Designer 里设置的 objectName 一模一样！
        self.btn_login.clicked.connect(self.on_login_click)

    # ================= 3. 写具体的功能 =================
    def on_login_click(self):
        # 提取输入框里的文字内容
        # 注意：input_username 也必须和 Designer 里的 objectName 一致
        user_name = self.input_username.text()

        # 做一个简单的判断功能
        if user_name == "":
            # 弹出一个警告框
            QMessageBox.warning(self, "警告", "名字不能为空哦！")
        else:
            # 弹出一个成功提示框
            QMessageBox.information(self, "成功", f"欢迎回来，{user_name}！")


# ================= 4. 程序的绝对启动开关 =================
if __name__ == '__main__':
    # 每一段 PyQt5 程序都必须有这四句，雷打不动
    app = QApplication(sys.argv)  # 启动系统引擎
    window = MyWindow()  # 实例化我们的窗口
    window.show()  # 把窗口显示在屏幕上
    sys.exit(app.exec_())  # 让程序一直运行，直到你点击右上角的 X