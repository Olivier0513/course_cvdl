from PyQt5 import QtWidgets
from hw1_ui import Ui_MainWindow
from pathlib import Path
import sys
import cv2
import numpy as np
from tkinter.filedialog import askdirectory
from PyQt5.QtWidgets import QFileDialog
import os
from os.path import join


class MainWindow(QtWidgets.QMainWindow):
    isCalibrate = False
    E = []

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        choices = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        self.ui.comboBox.addItems(choices)

        # ButtonEvents
        # 0. Load Img
        self.ui.pushButton_11.clicked.connect(self.LoadAllImg)
        self.ui.pushButton_12.clicked.connect(self.LoadImgL)
        self.ui.pushButton_13.clicked.connect(self.LoadImgR)

        # 1. Camera Calibration
        self.ui.pushButton_21.clicked.connect(self.CornerDetect)
        self.ui.pushButton_22.clicked.connect(self.FindIntrinsicMat)
        self.ui.pushButton_23.clicked.connect(self.FindExtrinsicMat)
        self.ui.pushButton_24.clicked.connect(self.FindDistortionMat)
        self.ui.pushButton_25.clicked.connect(self.UnDistort)

        # 2. Augmented Reality
        self.ui.pushButton_31.clicked.connect(self.ShowAR)
        self.ui.pushButton_32.clicked.connect(self.ShowARVertical)
        
        # 3. Stereo Disparity Map
        self.ui.pushButton_41.clicked.connect(self.Stereo)
        
        self.img = {
            "result": {
                "1": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "2": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "3": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "4": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "5": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "6": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "7": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "8": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "9": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "10": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "11": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "12": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "13": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "14": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp'),
                "15": cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp')
            }
        }

        folderPath = " "
        folderLPath = " "
        folderRPath = " "
        folderName = " "
        folderLName = " " 
        folderRName = " "
        fileCount = 0

    def LoadAllImg(self):
        global folderPath
        folderPath = QFileDialog.getExistingDirectory(self, "Open folder")
        global fileCount
        fileCount = 0

        for path in os.listdir(folderPath):
            if os.path.isfile(os.path.join(folderPath, path)) and "bmp" in path:
                fileCount += 1
        print(fileCount)        

    def LoadImgL(self):
        self.filenameL, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img_L = cv2.imread(self.filenameL)

    def LoadImgR(self):
        self.filenameR, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img_R = cv2.imread(self.filenameR) 
    
    def CornerDetect(self):
        global fileCount
        global folderPath
        global fileName
        for i in range(1,fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            self.img["result"][str(i)] = tempImg
            ret, corner = cv2.findChessboardCorners(self.img["result"][str(i)], (11, 8), None)
            if ret == True:
                cv2.drawChessboardCorners(self.img["result"][str(i)], (11, 8),corner, ret)

        for i in range(1,fileCount + 1):
            cv2.imshow("",cv2.resize(self.img["result"][str(i)], (900, 900)))
            cv2.waitKey(500)
            #time.sleep(1)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def FindIntrinsicMat(self):
        global fileCount
        global folderPath
        global fileName
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        
        for i in range(1, fileCount+1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            pic = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            ret, corner = cv2.findChessboardCorners(pic, (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corner)

        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        print("Intrinsic")
        print(mat)

    def FindExtrinsicMat(self):
        global fileCount
        global folderPath
        global fileName
        objp = np.zeros((11*8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        rets = []
        corners = []

        for i in range(1,fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            pic = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            ret, corner = cv2.findChessboardCorners(pic, (11, 8), None)
            rets.append(ret)
            corners.append(corner)
            objpoints.append(objp)
            imgpoints.append(corner)
        rets, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        pic_num = self.ui.comboBox.currentIndex()
        rotation = cv2.Rodrigues(rvec[pic_num])[0]

        print("Extrinsic:")
        print("[[{:f}, {:f}, {:f}, {:f}]".format(rotation[0][0], rotation[0][1], rotation[0][2], tvec[pic_num][0][0]))
        print("[{:f}, {:f}, {:f}, {:f}]".format(rotation[1][0], rotation[1][1], rotation[1][2], tvec[pic_num][1][0]))
        print("[{:f}, {:f}, {:f}, {:f}]]".format(rotation[2][0], rotation[2][1], rotation[2][2], tvec[pic_num][2][0]))
    
    def FindDistortionMat(self):
        global fileCount
        global folderPath
        global fileName
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        
        for i in range(1, fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            pic = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            ret, corner = cv2.findChessboardCorners(pic, (11, 8), None)           
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corner)
                
        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        with np.printoptions(precision = 6, suppress = True):
            print("Distortion:")
            print("[{}]".format(dist[0]))

    def UnDistort(self):
        global fileCount
        global folderPath
        global fileName
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        
        for i in range(1, fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            pic = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            ret, corner = cv2.findChessboardCorners(pic, (11, 8), None)           
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corner)
        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        for i in range(1, fileCount + 1):
            fileName = folderPath +"/"+ str(i) + ".bmp"
            tempImg = cv2.imread(fileName)
            self.img["result"][str(i)] = cv2.undistort(tempImg,mat,dist,None,mat)
            pic_show = np.hstack([cv2.resize(tempImg, (600, 600)),cv2.resize(self.img["result"][str(i)], (600, 600))])
            cv2.imshow("1_5",pic_show)
            cv2.waitKey(500)

    def ShowAR(self):
        global fileCount
        global folderPath
        global fileName
        # get library folder
        libraryPath = join(folderPath, 'Q2_lib', 'alphabet_lib_onboard.txt')
        fs = cv2.FileStorage(libraryPath,cv2.FILE_STORAGE_READ)

        # get textbox word  
        keyWord = self.ui.textbox.text()

        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        imgs = []

        for i in range(fileCount):
            fileName = folderPath +"/"+ str(i+1) + ".bmp"
            tempImg = cv2.imread(fileName)
            imgs.append(tempImg)
            ret, corners = cv2.findChessboardCorners(imgs[i], (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        
        # for each file
        for i in range(fileCount):
            for j in range(len(keyWord)):
                if j == 0:
                    dy = 5
                    dx = 7
                elif j == 1:
                    dy = 5
                    dx = 4           
                elif j == 2:
                    dy = 5
                    dx = 1
                elif j == 3:
                    dy = 2
                    dx = 7
                elif j == 4:
                    dy = 2
                    dx = 4
                elif j == 5:
                    dy = 2
                    dx = 1
                    
                ch = np.float32(fs.getNode(keyWord[j]).mat())    
                for k in range(ch.shape[0]):
                    ch[k][0][0] = ch[k][0][0] + dx
                    ch[k][0][1] = ch[k][0][1] + dy
                    ch[k][1][0] = ch[k][1][0] + dx
                    ch[k][1][1] = ch[k][1][1] + dy
                    ch[k] = np.float32(ch[k])
                    imgpoints,_ = cv2.projectPoints(ch[k],rvec[i],tvec[i],mat,dist)
                    imgs[i] = cv2.line(imgs[i],(int(imgpoints[0][0][0]),int(imgpoints[0][0][1])),(int(imgpoints[1][0][0]),int(imgpoints[1][0][1])), (0, 0, 255), 5)

        # show image
        cv2.namedWindow("AR", cv2.WINDOW_NORMAL)
        for i in range(0, fileCount):
            cv2.resizeWindow("AR", 600, 600)
            cv2.imshow("AR", imgs[i])
            cv2.waitKey(1000)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def ShowARVertical(self):
        global fileCount
        global folderPath
        global fileName
        # get library folder
        libraryPath = join(folderPath, 'Q2_lib', 'alphabet_lib_vertical.txt')
        fs = cv2.FileStorage(libraryPath,cv2.FILE_STORAGE_READ)

        # get textbox word  
        keyWord = self.ui.textbox.text()

        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        imgs = []
        
        for i in range(fileCount):
            fileName = folderPath +"/"+ str(i+1) + ".bmp"
            tempImg = cv2.imread(fileName)
            imgs.append(tempImg)
            ret, corners = cv2.findChessboardCorners(imgs[i], (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (1024, 1024), None, None)
        
        # for each file
        for i in range(fileCount):
            for j in range(len(keyWord)):
                if j == 0:
                    dy = 5
                    dx = 7
                elif j == 1:
                    dy = 5
                    dx = 4           
                elif j == 2:
                    dy = 5
                    dx = 1
                elif j == 3:
                    dy = 2
                    dx = 7
                elif j == 4:
                    dy = 2
                    dx = 4
                elif j == 5:
                    dy = 2
                    dx = 1

                ch = np.float32(fs.getNode(keyWord[j]).mat())    
                for k in range(ch.shape[0]):
                    ch[k][0][0] = ch[k][0][0] + dx
                    ch[k][0][1] = ch[k][0][1] + dy
                    ch[k][1][0] = ch[k][1][0] + dx
                    ch[k][1][1] = ch[k][1][1] + dy
                    ch[k] = np.float32(ch[k])
                    imgpoints,_ = cv2.projectPoints(ch[k],rvec[i],tvec[i],mat,dist)
                    imgs[i] = cv2.line(imgs[i],(int(imgpoints[0][0][0]),int(imgpoints[0][0][1])),(int(imgpoints[1][0][0]),int(imgpoints[1][0][1])), (0, 0, 255), 5)

        # show image
        cv2.namedWindow("AR", cv2.WINDOW_NORMAL)
        for i in range(0, fileCount):
            cv2.resizeWindow("AR", 600, 600)
            cv2.imshow("AR", imgs[i])
            cv2.waitKey(1000)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def Stereo(self):
        # save original image 
        self.img_L_org = self.img_L
        self.img_R_org = self.img_R
        # image gray
        self.img_L = cv2.cvtColor(self.img_L, cv2.COLOR_BGR2GRAY)
        self.img_R = cv2.cvtColor(self.img_R, cv2.COLOR_BGR2GRAY)
        # find the disparity image
        stereoBM = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = cv2.normalize(stereoBM.compute(self.img_L, self.img_R), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.disparity_org = stereoBM.compute(self.img_L, self.img_R)

        cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
        cv2.imshow("disparity", self.disparity)
        cv2.resizeWindow("disparity", 1000, 680)

        event = cv2.EVENT_LBUTTONDOWN
        cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
        cv2.namedWindow('imgR_dot', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('imgL', self.drawCircle)
        cv2.resizeWindow("imgL", 1000, 680)
        cv2.resizeWindow("imgR_dot", 1000, 680)
        cv2.imshow('imgL',self.img_L_org)
        cv2.imshow('imgR_dot',self.img_R_org)
        cv2.waitKey(0)

    def drawCircle(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            d = self.disparity_org[y][x]/16
            if d > 0:
                img_R_dot = np.copy(self.img_R_org)
                cv2.circle(img_R_dot, (int(x-d), y), 10, (0, 255, 0), -1)
                cv2.imshow('imgR_dot',img_R_dot)
            else:
                img_L_dot = np.copy(self.img_L_org)
                cv2.circle(img_L_dot, (x, y), 10, (0, 0, 255), -1)
                cv2.imshow('imgL',img_L_dot)
            

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
