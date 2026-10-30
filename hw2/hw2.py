from PyQt5 import QtWidgets
from hw2_ui import Ui_MainWindow
from pathlib import Path
import sys
import cv2
import numpy as np
from tkinter.filedialog import askdirectory
from PyQt5.QtWidgets import QFileDialog
import os
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # 0
        self.ui.loadVideo.clicked.connect(self.loadVideo)
        self.ui.loadImage.clicked.connect(self.loadImage)
        self.ui.loadFolder.clicked.connect(self.loadFolder)
        # 1. Background Subtraction
        self.ui.backgroundSub.clicked.connect(self.backgroundSub)
        # 2. Optical Flow
        self.ui.preprocessing.clicked.connect(self.preprocessing)
        self.ui.videoTracking.clicked.connect(self.videoTracking)
        # 3. Perspective Transform
        self.ui.perspectiveTrans.clicked.connect(self.perspectiveTrans)
        # 4. PCA
        self.ui.imageReconstruct.clicked.connect(self.imageReconstruct)
        self.ui.computeError.clicked.connect(self.computeError)

    def loadVideo(self):
        self.videoFileName, _ = QFileDialog.getOpenFileName(self, "Open file", "./")

    def loadImage(self):
        self.imgFileName, _ = QFileDialog.getOpenFileName(self, "Open file", "./")      

    def loadFolder(self):
        self.folderDirectory = QFileDialog.getExistingDirectory(self,"Open folder", "./")
    
    # 1. Background Subtraction
    def backgroundSub(self):
        # open video 
        cap = cv2.VideoCapture(self.videoFileName)
        first_25_frames = []
        count = 0
        # Get first 25 frames
        if cap.isOpened() == False:
            print("Cannot open video.")
        else:
            while True:
                ret, frame = cap.read()
                if ret == True:
                    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    first_25_frames.append(gray_frame)
                count+=1
                if count > 24:
                    break
        
        cap.release()

        # Get mean and std
        first_25_array = np.array(first_25_frames)
        mean_array = first_25_array.mean(axis=0)
        std_array = first_25_array.std(axis=0)
        std_array[np.where(std_array<5)] = 5
                    
        cap = cv2.VideoCapture(self.videoFileName)
        if cap.isOpened() == False:
            print("Cannot open video.")
        else:
            while True:
                ret, frame = cap.read()
                if ret == True:
                    foreground = frame.copy()
                    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    h = frame.shape[0]
                    w = frame.shape[1]
                    mask = np.zeros([h,w,3])
                    # mask                  
                    mask[:,:,0][np.where(abs(gray_frame - mean_array) > std_array * 5)] = 255
                    mask[:,:,1][np.where(abs(gray_frame - mean_array) > std_array * 5)] = 255
                    mask[:,:,2][np.where(abs(gray_frame - mean_array) > std_array * 5)] = 255
                    # foreground
                    foreground[np.where(mask==0)] = 0

                    result = np.zeros([h,w*3,3])
                    result[0:h,0:w,:] = frame/255
                    result[0:h,w:2*w,:] = mask/255
                    result[0:h,2*w:3*w,:] = foreground/255
                    cv2.imshow('1. Background Subtraction',result)
                    key = cv2.waitKey(16)
                    
                else:
                    break
        cap.release()

    # 2. Optical Flow
    def preprocessing(self): 
        cap = cv2.VideoCapture(self.videoFileName)
        if cap.isOpened() == False:
            print("Cannot open video.")
        else:
            ret, frame = cap.read()
        cap.release()

        params = cv2.SimpleBlobDetector_Params()
        # Set properties
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 90
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        # Create detector and check blobs
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(frame)

        self.keypoints_list = []
        for i in range(len(keypoints)):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])
            self.keypoints_list.append([[x,y]])
            frame_keypoints = cv2.line(frame,(x,y-6),(x,y+6),(0,0,255))
            frame_keypoints = cv2.line(frame,(x-6,y),(x+6,y),(0,0,255))
            frame_keypoints = cv2.rectangle(frame,(x-6,y-6),(x+6,y+6),(0,0,255))
        cv2.imshow('2.1 Preprocessing', frame_keypoints)

    def videoTracking(self): 
        cap = cv2.VideoCapture(self.videoFileName)
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        lk_params = dict( winSize=(15,15),
                          maxLevel=2,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03) )
        # Track points
        p0 = np.array(self.keypoints_list,dtype=np.float32)
        # Draw line
        mask = np.zeros_like(old_frame)

        if cap.isOpened() == False:
            print("Cannot open video.")
        else:
            while True:
                ret, frame = cap.read()    
                if ret == True:
                    new_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                    # Calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    # Draw the points' track
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
                        frame = cv2.circle(frame, (int(a), int(b)), 5, (0,0,255), -1)

                    img = cv2.add(frame, mask)
                    cv2.imshow('2.2 Video Tracking', img)
                    cv2.waitKey(30)

                    # Update previous frame and points
                    old_gray = new_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                    
                else:
                    break
        cap.release()

    # 3. Perspective Transform
    def perspectiveTrans(self): 
        cap = cv2.VideoCapture(self.videoFileName)
        projectImg = cv2.imread(self.imgFileName)

        project_h,project_w,c = projectImg.shape
        projectList = [ (0, 0), (project_w, 0),(project_w, project_h), (0, project_h)]

        if cap.isOpened() == False:
            print("Cannot open video.")
        else:
            while True:
                ret, frame = cap.read()
                if ret == True:
                    h,w,c = frame.shape
                    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                    arucoDictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
                    arucoParameters = cv2.aruco.DetectorParameters_create()
                    corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, arucoDictionary, parameters = arucoParameters)

                    if len(corners)==4:
                        count=0
                        ids = ids.flatten()
                        for (markerCorner, markerID) in zip(corners, ids):
                            corners = markerCorner.reshape((4, 2))
                            topLeft, topRight, bottomRight, bottomLeft = corners

                            topRight = ( int(topRight[0]), int(topRight[1]) )
                            bottomRight = ( int(bottomRight[0]), int(bottomRight[1]) )
                            bottomLeft = ( int(bottomLeft[0]), int(bottomLeft[1]) )
                            topLeft = ( int(topLeft[0]), int(topLeft[1]) )    
                            # 1  2
                            # 4  3                                               
                            if(ids[count]==1):  
                                pt0 = topLeft
                            elif(ids[count]==2):
                                pt1 = topRight
                            elif(ids[count]==3):
                                pt2 = bottomRight
                            else:
                                pt3 = bottomLeft
                            count = count + 1

                        pts_dst = np.array([pt0,pt1,pt2,pt3])
                        homo, status = cv2.findHomography(np.float32(projectList), np.float32(pts_dst))
                        im_out = cv2.warpPerspective(projectImg, homo, (frame.shape[1],frame.shape[0]))

                        img2gray = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
                        ret, mask = cv2.threshold(img2gray, 0, 255 ,cv2.THRESH_BINARY)
                        mask_inv = cv2.bitwise_not(mask)
                        img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                        dst = cv2.add(img1_bg, im_out)

                        concatImg = cv2.hconcat([frame, dst])
                        concatImg = cv2.resize(concatImg, (int(w*0.8), int(h*0.4)))
                        
                        cv2.imshow('3. Perspective Transform', concatImg)
                        cv2.waitKey(30)

                else:
                    break

    # 4. PCA
    def imageReconstruct(self): 
        # Get all directories' name
        dirs = listdir(self.folderDirectory)

        # Store all images
        self.originalImg = []
        for i in range(len(dirs)):
            img_path = join(self.folderDirectory,dirs[i])
            img = cv2.imread(img_path)
            self.originalImg.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        # Reconstruct image
        imgNum = len(self.originalImg)
        h,w,c = self.originalImg[0].shape
        data = np.array(deepcopy(self.originalImg)).reshape(imgNum,-1)

        pca = PCA(n_components = int(imgNum*0.8))
        reducedImg = pca.fit_transform(data)
        self.reconstructedImg = pca.inverse_transform(reducedImg)

        self.reconstructedImg = list(self.reconstructedImg.reshape(imgNum,h,w,c))
        for i in range(imgNum):
            self.reconstructedImg[i] =cv2.normalize(self.reconstructedImg[i], self.reconstructedImg[i], alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype = cv2.CV_8UC1 )
        
        # Draw output
        fig, ax = plt.subplots(nrows=4, ncols=15, figsize=(30,8))
        fig.tight_layout()

        count1 = 0
        count2 = 0

        for i, axi in enumerate(ax.flat):
            rowid = i // 15
            if rowid % 2 == 0:
                img = self.originalImg[count1]
                count1+=1
            else:
                img = self.reconstructedImg[count2]
                count2+=1
            
            axi.set_axis_off()
            axi.imshow(img)

        plt.axis('off')
        plt.show()

    def computeError(self):
        # Compute error 
        errs = []
        for img1,img2 in zip(self.originalImg,self.reconstructedImg): 
            ori_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            rc_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            err = np.sqrt(np.sum((ori_gray-rc_gray)**2))
            errs.append(int(err))

        # Print out max and min
        errs.sort()
        print('reconstruction error:')
        print('max error: ',errs[len(errs)-1])
        print('min error: ',errs[0])

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
