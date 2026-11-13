# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2_05.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(350, 590)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 311, 521))
        self.groupBox.setObjectName("groupBox")
        self.loadImg = QtWidgets.QPushButton(self.groupBox)
        self.loadImg.setGeometry(QtCore.QRect(30, 40, 251, 61))
        self.loadImg.setObjectName("loadImg")
        self.showImg = QtWidgets.QPushButton(self.groupBox)
        self.showImg.setGeometry(QtCore.QRect(30, 120, 251, 61))
        self.showImg.setObjectName("showImg")
        self.showDistribution = QtWidgets.QPushButton(self.groupBox)
        self.showDistribution.setGeometry(QtCore.QRect(30, 200, 251, 61))
        self.showDistribution.setObjectName("showDistribution")
        self.showModel = QtWidgets.QPushButton(self.groupBox)
        self.showModel.setGeometry(QtCore.QRect(30, 280, 251, 61))
        self.showModel.setObjectName("showModel")
        self.showCompare = QtWidgets.QPushButton(self.groupBox)
        self.showCompare.setGeometry(QtCore.QRect(30, 360, 251, 61))
        self.showCompare.setObjectName("showCompare")
        self.inference = QtWidgets.QPushButton(self.groupBox)
        self.inference.setGeometry(QtCore.QRect(30, 440, 251, 61))
        self.inference.setObjectName("inference")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 350, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CvDl 2022 HW2 Q5"))
        self.groupBox.setTitle(_translate("MainWindow", "5. ResNet50"))
        self.loadImg.setText(_translate("MainWindow", "Load Image"))
        self.showImg.setText(_translate("MainWindow", "1. Show Images"))
        self.showDistribution.setText(_translate("MainWindow", "2. Show Distribution"))
        self.showModel.setText(_translate("MainWindow", "3. Show Model Structure"))
        self.showCompare.setText(_translate("MainWindow", "4. Show Comparison"))
        self.inference.setText(_translate("MainWindow", "5. Inference"))

