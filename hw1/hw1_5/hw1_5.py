from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from hw1_5_ui import Ui_MainWindow

import sys

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
from torchsummary import summary

from random import randrange

import matplotlib.pyplot as plt

class VGG19(torch.nn.Module):
    def __init__(self,num_classes):
        super(VGG19, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block5  = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = self.classifier(x.view(-1,512*1*1))
        probas = F.softmax(logits,dim = 1)
        return logits,probas

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_1.clicked.connect(self.LoadImg)
        self.ui.pushButton_2.clicked.connect(self.ShowImg)
        self.ui.pushButton_3.clicked.connect(self.ShowModel)
        self.ui.pushButton_4.clicked.connect(self.ShowAugmentation)
        self.ui.pushButton_5.clicked.connect(self.ShowAccuracy)
        self.ui.pushButton_6.clicked.connect(self.Inference)

        self.train_dataset = torchvision.datasets.CIFAR10(root='.', train=True, transform=transforms.ToTensor(), download=True)

        (self.traindata,self.valdata) = random_split(self.train_dataset,[40000,10000],generator=torch.Generator().manual_seed(42))

    def LoadImg(self):
        self.filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        self.img = Image.open(self.filename).convert("RGB")
        
    def ShowImg(self):
        text = ['airplane', 'automobile', 'bird' ,'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        #(x_train, y_train), (x_test, y_test) = train_dataset.load_data()
        plt.figure(figsize=(8,8),facecolor='w')
        for i in range(3):
          for j in range(3):
            index = randrange(len(self.traindata))
            img, label = self.traindata[index]
            plt.subplot(3, 3, i*3+j+1)
            plt.title(text[label])
            plt.imshow(img.permute(1,2,0))
            plt.axis('off')

        plt.show()     

    def ShowModel(self):
        model = VGG19(10)
        # model.load_state_dict(torch.load(f="./vgg19_model.pth",map_location=torch.device('cpu') ))
        model.eval()
        summary(model, (3, 32, 32))

    def ShowAugmentation(self):
        if not hasattr(self, 'img'):
            print("Please load an image first.")
            return

        plt.figure(figsize=(8,4),facecolor='w')

        transform_1 = transforms.Compose([transforms.ToTensor(),transforms.RandomRotation(180, expand=True, center=None)])
        img_1 = transform_1(self.img).permute(1,2,0).numpy()
        #plt.figure()
        plt.subplot(1,3,1)
        plt.title("RandomRotation")
        plt.imshow(img_1)
        plt.axis('off')
        

        transform_2 = transforms.Compose([transforms.ToTensor(),transforms.RandomResizedCrop(size = (200,200),scale=(0.2, 1.0), ratio=(0.5, 1.1))])
        img_2 = transform_2(self.img).permute(1,2,0).numpy()
        #plt.figure()
        plt.subplot(1,3,2)
        plt.title("RandomResizedCrop")
        plt.imshow(img_2)
        plt.axis('off')
        

        transform_3 = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(p=0.9)])
        img_3 = transform_3(self.img).permute(1,2,0).numpy()
        #plt.figure()
        plt.subplot(1,3,3)
        plt.title("RandomHorizontalFlip")
        plt.imshow(img_3)
        plt.axis('off')

        plt.show()           
    
    def ShowAccuracy(self):
        img1 = cv2.imread('accuracy.png')
        img2 = cv2.imread('loss.png')
        plt.figure()
        plt.imshow(img1)
        plt.axis('off')
        plt.show()
        # img2 = cv2.imread('loss.png')
        plt.figure()
        plt.imshow(img2)
        plt.axis('off')
        plt.show()
        

    def Inference(self):
        if not hasattr(self, 'img') or self.img is None:
            print("Please load an image first.")
            return
        
        model = VGG19(10)
        model.load_state_dict(torch.load(f="./vgg19_model.pth",map_location=torch.device('cpu') ))
        model.eval()
        #image = Image.open('0.jpg')
        x = TF.to_tensor(self.img)
        x.unsqueeze_(0)
        with torch.inference_mode():
            predict = model(x)
            text = ['airplane', 'automobile', 'bird' ,'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            print(text[predict[1].argmax()])
            print(predict[1].max().item())
            plt.figure()
            plt.imshow(self.img)
            plt.axis('off')
            plt.title(f"Confidence = {predict[1].max().item():.4f}\nPrediction Label = {text[predict[1].argmax()]}")
            plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
