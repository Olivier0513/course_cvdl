from PyQt5 import QtWidgets
from hw1_4_ui import Ui_MainWindow
from pathlib import Path
import sys
import cv2
import numpy as np
from tkinter.filedialog import askdirectory
from PyQt5.QtWidgets import QFileDialog
import os
from os.path import join
import matplotlib.pyplot as plt


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_1.clicked.connect(self.LoadImg_1)
        self.ui.pushButton_2.clicked.connect(self.LoadImg_2)
        self.ui.pushButton_3.clicked.connect(self.Keypoint)
        self.ui.pushButton_4.clicked.connect(self.Match)

    def LoadImg_1(self):
        self.filename1, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img_1 = cv2.imread(self.filename1)
        self.img_gray_1 = cv2.cvtColor(self.img_1,cv2.COLOR_BGR2GRAY)

    def LoadImg_2(self):
        self.filename2, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img_2 = cv2.imread(self.filename2)  
        self.img_gray_2 = cv2.cvtColor(self.img_2,cv2.COLOR_BGR2GRAY)      

    def Keypoint(self):
        # Scale-Invariant Feature Transform
        sift = cv2.SIFT_create()
        keypoints = sift.detect(self.img_gray_1, None)
        
        result = cv2.drawKeypoints(self.img_gray_1, keypoints, 0, (0,255,0), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        plt.figure()
        plt.imshow(result)
        plt.axis('off')
        plt.title("Figure1")
        plt.show()

    def Match(self): 
        sift = cv2.SIFT_create()
        key1, des1 = sift.detectAndCompute(self.img_gray_1, None)
        key2, des2 = sift.detectAndCompute(self.img_gray_2, None)

        # draw1 = cv2.drawKeypoints(self.img_gray_1, key1, 0, (0,255,0), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # draw2 = cv2.drawKeypoints(self.img_gray_2, key2, 0, (0,255,0), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # matchNum = int(self.ui.textbox.text())

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = [[0, 0] for i in range(len(matches))]

        # Lowe's Ratio Test
        for i, pair in enumerate(matches):
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.5 * n.distance:
                    good_matches[i] = [1, 0]

        # Draw the matched keypoints
        # result = cv2.drawMatches(draw1,key1,draw2,key2,matches[:matchNum],None,matchColor =(255,255,0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        result = cv2.drawMatchesKnn(self.img_gray_1, key1, self.img_gray_2, key2, matches, None, matchColor=(255,255,0), singlePointColor=(0, 255, 0), matchesMask=good_matches, flags=0)

        plt.figure()
        plt.imshow(result)
        plt.axis('off')
        plt.title("Figure2")
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
