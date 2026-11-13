from PyQt5 import QtWidgets
from hw2_05_ui import Ui_MainWindow
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

import torch
from torchvision.models import resnet50
import torch.nn as nn
from torchsummary import summary
import random
from PIL import Image
from torchvision import datasets,transforms,ops
from torchvision.transforms import ToTensor, Resize


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.loadImg.clicked.connect(self.LoadImg)
        self.ui.showImg.clicked.connect(self.ShowImg)
        self.ui.showDistribution.clicked.connect(self.ShowDistribution)
        self.ui.showModel.clicked.connect(self.ShowModel)
        self.ui.showCompare.clicked.connect(self.ShowCompare)
        self.ui.inference.clicked.connect(self.Inference)

        self.transform = transforms.Compose([
                            Resize(size=(224, 224)),
                            ToTensor() 
                        ])

    def LoadImg(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.input_img = Image.open(filename).convert("RGB")

    def ShowImg(self):
        dir1 = listdir('./inference_dataset/Cat')
        catImg = []
        for i in range(len(dir1)):
            img_path = join('./inference_dataset/Cat',dir1[i])
            img = Image.open(img_path).convert("RGB")
            catImg.append(img)

        dir2 = listdir('./inference_dataset/Dog')
        dogImg = []
        for i in range(len(dir2)):
            img_path = join('./inference_dataset/Dog',dir2[i])
            img = Image.open(img_path).convert("RGB")
            dogImg.append(img)

        random_cat = catImg[random.randint(0,len(dir1)-1)]
        random_dog = dogImg[random.randint(0,len(dir2)-1)]
        random_cat = self.transform(random_cat).permute(1,2,0).numpy()
        random_dog = self.transform(random_dog).permute(1,2,0).numpy()

        plt.subplot(1,2,1)
        plt.title('Cat')
        plt.imshow(random_cat)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,2,2)
        plt.title('Dog')
        plt.imshow(random_dog)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def ShowDistribution(self):
        print('')
        # Calculate Number
        # dir1 = './Dataset_OpenCvDl_Hw2_Q5/training_dataset/Cat'
        # cat = len([name for name in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, name))])
        # dir2 = './Dataset_OpenCvDl_Hw2_Q5/training_dataset/Dog'
        # dog = len([name for name in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, name))])
        # # Plot the figure
        # plt.ylabel("Number of images")
        # plt.title("Class Distribution")
        # bar = plt.bar(['Cat','Dog'],[cat,dog]) 
        # plt.text(   bar[0].get_x()+bar[0].get_width()/2.,
        #             cat,
        #             "%d" % int(cat),
        #             ha="center",
        #             va = "bottom",  )
        # plt.text(   bar[1].get_x()+bar[1].get_width()/2.,
        #             dog,
        #             "%d" % int(dog),
        #             ha="center",
        #             va = "bottom", )
        # plt.show()
        # # Show Image
        img = cv2.imread("./figure/distribution.png")
        cv2.imshow('Class Distribution',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def ShowModel(self):
        model = resnet50(weights=None)
        model.fc = nn.Linear(2048,1)
        model = nn.Sequential(model,nn.Sigmoid())
        summary(model, (3, 224, 224))

    def ShowCompare(self):
        # plt.ylabel("Accuracy (%)")
        # plt.title("Accuracy Comparison")
        # bar = plt.bar(['Binary Cross Entropy','Focal loss'],[88.27,89.64]) 
        # plt.text(   bar[0].get_x()+bar[0].get_width()/2.,
        #             88.27,
        #             "%.2f" % float(88.27),
        #             ha="center",
        #             va = "bottom",  )
        # plt.text(   bar[1].get_x()+bar[1].get_width()/2.,
        #             89.64,
        #             "%.2f" % float(89.64),
        #             ha="center",
        #             va = "bottom", )
        # plt.show()
        img = cv2.imread("./figure/comparison.png")
        cv2.imshow('Accuracy Comparison',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Inference(self):
        img = self.transform(self.input_img)
        img.unsqueeze_(0)
        print(img.shape)
        model = resnet50(weights=None)
        model.fc = nn.Linear(2048,1)
    
        model.load_state_dict(torch.load(f="./model/resnet50_focal_model.pth",map_location=torch.device('cpu') ))
        model.eval()
        
        with torch.inference_mode():
            predict = model(img)
            sigmoid = nn.Sigmoid()
            predict_result = torch.round(sigmoid(predict))
            if predict_result == 0:
                # cat
                result = 'Cat'
                print('cat')
            elif predict_result == 1:
                # dog
                result = 'Dog'
                print('dog')

            plt.figure()
            plt.imshow(img.squeeze_(0).permute(1,2,0).numpy())
            plt.axis('off')
            plt.title(f"Prediction: {str(result)}")
            plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
