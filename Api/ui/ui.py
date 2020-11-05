from kiwoom.kiwoom import *
from PyQt5.QtWidgets import *
import sys


class Ui_class:
    def __init__(self):
        print("UI class")

        # QApplication 은 app 에 대한 초기화를 해준다.
        # sys.argv 는 경로, mode, port 가 담겨있는 리스트다.
        self.app = QApplication(sys.argv)

        self.kiwoom = Kiwoom()

        self.app.exec_()  # 이벤트 루프를 실행시킴
