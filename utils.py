'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
# main file
'''Train CIFAR10 with PyTorch.'''
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from PytorchProject.models.resnet import *


import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


class data_utils:
    
    def __init__(self):
        # self.best_acc = 0  # best test accuracy
        # self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        # self.args = args
        pass
    
    def show_misclassified_images( self, misc_im,misc_tr,misc_pred):
        self.misc_im=misc_im.cpu()
        self.misc_tr=misc_tr.cpu()
        self.misc_pred=misc_pred.cpu()
        fig=plt.figure(figsize=(4, 10))
        columns = 2
        rows = 5
        for i in range(1, columns*rows +1):
            img = self.misc_im[i-1]
            print('img is ', img)
            img = torch.transpose(img, 0, 2)
            print('img2 is ', img)
            p = self.misc_pred[i-1]
            t = self.misc_tr[i-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img) 
            plt.axis('off')
            plt.title("Pred:"+str(p)[7:8]+"  Act: "+str(t)[7:8])
        plt.show()

    def plot_accuracy_loss_graph(self, train_loss_graph, train_accuracy_graph, test_loss_graph, test_accuracy_graph):
        self.train_loss_graph =train_loss_graph
        self.train_accuracy_graph=train_accuracy_graph
        self.test_loss_graph = test_loss_graph 
        self.test_accuracy_graph=test_accuracy_graph
        plt.figure(figsize=(10,5))
        plt.title("Test Loss and Train Loss")
        plt.plot(self.test_loss_graph,label="Test Loss")
        plt.plot(self.train_loss_graph,label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
        plt.figure(figsize=(10,5))
        plt.title("Train Accuracy and Test Accuracy")
        plt.plot(self.test_accuracy_graph,label="Test Accuracy")
        plt.plot(self.train_accuracy_graph,label="Train Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def data(self, dataset, batch_size):
        self.batch_size = batch_size
        print('==> Preparing data..')
        self.dataset = dataset
        if self.dataset == 'CIFAR10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            self.trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=self.transform_train)
            self.trainloader = torch.utils.data.DataLoader(
                self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
    
            self.testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=self.transform_test)
            self.testloader = torch.utils.data.DataLoader(
                self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
    
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            print('unknown dataset provided by user.')

        return self.trainloader, self.testloader