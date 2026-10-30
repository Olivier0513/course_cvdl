# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(300, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(5, 10, 290, 550))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_1 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_1.setGeometry(QtCore.QRect(55, 70, 181, 51))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(55, 150, 181, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(55, 230, 181, 51))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(55, 310, 181, 51))
        self.pushButton_4.setObjectName("pushButton_4")
        # self.textbox = QtWidgets.QLineEdit(self.groupBox)
        # self.textbox.setGeometry(QtCore.QRect(100, 370, 100, 25))
        # self.textbox.setObjectName("textbox")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "4. SIFT"))
        self.pushButton_1.setText(_translate("MainWindow", "Load Image 1"))
        self.pushButton_2.setText(_translate("MainWindow", "Load Image 2"))
        self.pushButton_3.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.pushButton_4.setText(_translate("MainWindow", "4.2 Matched Keypoints"))

