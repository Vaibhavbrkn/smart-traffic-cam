from PyQt5.QtWidgets import QLabel, QPushButton, QApplication, QMainWindow, QFrame, QComboBox, QFileDialog
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont
import sys

from concurrent.futures import ThreadPoolExecutor
import time
from license.func import license_func
from func import tracking_func


class myWindow(QMainWindow):

    def __init__(self):
        super(myWindow, self).__init__()
        self.setGeometry(430, 150, 585, 440)
        self.setWindowTitle('Video Selection Tool')
        self.initUI()

    def initUI(self):
        self.path1 = None
        self.path2 = None
        self.path3 = None
        self.path4 = None
        self.checkVideo = 0

        self.line_1 = QFrame(self)
        self.line_1.setGeometry(QRect(30, 10, 521, 20))
        self.line_1.setFrameShape(QFrame.HLine)

        self.line_2 = QFrame(self)
        self.line_2.setGeometry(QRect(20, 20, 20, 391))
        self.line_2.setFrameShape(QFrame.VLine)

        self.line_3 = QFrame(self)
        self.line_3.setGeometry(QRect(540, 20, 20, 391))
        self.line_3.setFrameShape(QFrame.VLine)

        self.line_4 = QFrame(self)
        self.line_4.setGeometry(QRect(30, 400, 521, 20))
        self.line_4.setFrameShape(QFrame.HLine)

        self.label = QLabel(self)
        self.label.setText("Smart Traffic Camera")
        self.label.move(210, 40)
        self.label.setFont(QFont('MS Shell Dlg 2', 12, QFont.Bold))
        self.label.adjustSize()

        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(QRect(220, 90, 151, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Select Operation")
        self.comboBox.addItem("Traffic scheduling")
        self.comboBox.addItem("Number Plate Detection")

        self.video_1 = QPushButton(self)
        self.video_1.setText('Video 1')
        self.video_1.setGeometry(QRect(120, 170, 141, 31))
        self.video_1.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";")
        self.video_1.clicked.connect(self.getPath1)

        self.video_2 = QPushButton(self)
        self.video_2.setText('Video 2')
        self.video_2.setGeometry(QRect(320, 170, 141, 31))
        self.video_2.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";")
        self.video_2.clicked.connect(self.getPath2)

        self.video_3 = QPushButton(self)
        self.video_3.setText('Video 3')
        self.video_3.setGeometry(QRect(120, 240, 141, 31))
        self.video_3.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";")
        self.video_3.clicked.connect(self.getPath3)

        self.video_4 = QPushButton(self)
        self.video_4.setText('Video 4')
        self.video_4.setGeometry(QRect(320, 240, 141, 31))
        self.video_4.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";")
        self.video_4.clicked.connect(self.getPath4)

        self.runButton = QPushButton(self)
        self.runButton.setText("RUN")
        self.runButton.setGeometry(QRect(220, 320, 151, 31))
        self.runButton.setStyleSheet("font: 75 11pt \"MS Shell Dlg 2\";\n""")
        self.runButton.clicked.connect(self.checkOperation)

        self.errorLabel = QLabel(self)
        self.errorLabel.setText("Either operation or video not selected!!")
        self.errorLabel.move(204, 370)
        self.errorLabel.setStyleSheet('color: red')
        self.errorLabel.adjustSize()
        self.errorLabel.setHidden(True)

    def getPath1(self):
        self.path1 = QFileDialog.getOpenFileName()[0]
        self.checkVideo = self.checkVideo + 1
        print(self.path1)

    def getPath2(self):
        self.path2 = QFileDialog.getOpenFileName()[0]
        self.checkVideo = self.checkVideo + 1
        print(self.path2)

    def getPath3(self):
        self.path3 = QFileDialog.getOpenFileName()[0]
        self.checkVideo = self.checkVideo + 1
        print(self.path3)

    def getPath4(self):
        self.path4 = QFileDialog.getOpenFileName()[0]
        self.checkVideo = self.checkVideo + 1
        print(self.path4)

    def checkOperation(self):

        if (self.comboBox.currentIndex() != 0):
            if(self.checkVideo):
                self.fetchResult()
                self.errorLabel.setHidden(True)
            else:
                self.errorLabel.setHidden(False)
        else:
            self.errorLabel.setHidden(False)

    def fetchResult(self):

        print(
            f'{self.comboBox.currentIndex()}, {self.path1}, {self.path2}, {self.path3}, {self.path4}')
        print("hello")

        if self.comboBox.currentIndex() == 1:
            executors_list = []

            with ThreadPoolExecutor(max_workers=5) as executor:
                executors_list.append(executor.submit(
                    tracking_func, 'tf', './checkpoints/yolov4-tiny-416', 416, True, 'yolov4', self.path1, './outputs/tiny.avi', 'XVID', 0.45, 0.50, False, False, True, 'window1', 'cam1.txt', 'cam2.txt', 'out1.txt', './jumpes/1'))
                executors_list.append(executor.submit(tracking_func, 'tf', './checkpoints/yolov4-tiny-416', 416, True,
                                                      'yolov4', self.path1, './outputs/tiny.avi', 'XVID', 0.45, 0.50, False, False, True, 'window2', 'cam3.txt', 'cam4.txt', 'out2.txt', './jumpes/2'))

            for x in executors_list:
                print(x.result())

        else:
            executors_list = []

            with ThreadPoolExecutor(max_workers=5) as executor:
                executors_list.append(executor.submit(
                    license_func, 'tf', './license/checkpoints/custom-416', 416, False, 'yolov4', self.path1, './license/detections/gui.avi', 'XVID', 0.45, 0.50, False, False, False, False, True))

            for x in executors_list:
                print(x.result())


def window():
    app = QApplication(sys.argv)
    win = myWindow()
    win.show()
    sys.exit(app.exec_())


window()
