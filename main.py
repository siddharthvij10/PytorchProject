# main file
'''Train CIFAR10 with PyTorch.'''
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

class network_sample:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    def __init__(self, args):
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.args = args
        
    """def data(self, dataset, batch_size):
        self.batch_size = batch_size
        print('==> Preparing data..')
        self.dataset = dataset
        if self.dataset == 'CIFAR10':
            self.transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
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
            print('unknown dataset provided by user.')"""

    
    def model(self, opt, sch):
        self.opt = opt
        self.sch = sch
        print('==> Building model..')
        self.net = ResNet18()
        self.net = self.net.to(self.device)
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        
        if self.opt == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args,
                  momentum=0.9, weight_decay=5e-4)
        if self.sch == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        return self.net


    # Training
    def train(self, epoch, trainloader, device, train_loss_graph, train_accuracy_graph):
        self.epoch = epoch
        self.trainloader = trainloader
        self.device = device
        self.train_loss_graph = train_loss_graph
        self.train_accuracy_graph = train_accuracy_graph
        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        self.train_loss = 0
        self.correct = 0
        self.total = 0
        for self.batch_idx, (self.inputs, self.targets) in enumerate(self.trainloader):
            self.inputs, self.targets = self.inputs.to(self.device), self.targets.to(self.device)
            self.optimizer.zero_grad()
            self.outputs = self.net(self.inputs)
            self.loss = self.criterion(self.outputs, self.targets)
            self.loss.backward()
            self.optimizer.step()
    
            self.train_loss += self.loss.item()
            _, self.predicted = self.outputs.max(1)
            self.total += self.targets.size(0)
            self.correct += self.predicted.eq(self.targets).sum().item()
    
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        self.train_loss /= len(self.trainloader.dataset)
        self.accuracy = 100. * self.correct / len(self.trainloader.dataset)
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.train_loss, self.correct, len(self.trainloader.dataset), self.accuracy
            ))
        
        self.train_loss_graph.append(self.train_loss)
        self.train_accuracy_graph.append(self.accuracy)
        return self.train_loss_graph, self.train_accuracy_graph, self.accuracy
        
        
    def test(self, epoch, testloader, device, test_loss_graph, test_accuracy_graph):
        # global self.best_acc
        self.epoch = epoch 
        self.testloader = testloader
        self.device = device
        self.test_loss_graph = test_loss_graph 
        self.test_accuracy_graph = test_accuracy_graph
        self.net.eval()
        self.test_loss = 0
        self.correct = 0
        self.total = 0
        self.flg = 0
        with torch.no_grad():
            for self.batch_idx, (self.inputs, self.targets) in enumerate(self.testloader):
                self.inputs, self.targets = self.inputs.to(self.device), self.targets.to(self.device)
                self.outputs = self.net(self.inputs)
                self.loss = self.criterion(self.outputs, self.targets)
    
                self.test_loss += self.loss.item()
                _, self.predicted = self.outputs.max(1)
                self.total += self.targets.size(0)
                self.correct += self.predicted.eq(self.targets).sum().item()
    
                # Saving misclassified Images and their actual and pedicted labels
                self.tgt = self.targets.view_as(self.predicted)
                self.comp_df= self.predicted.eq(self.tgt)
                self.mis_c = ~self.comp_df
                if self.flg == 0:
                    self.misc_im = self.inputs[self.mis_c]
                    self.misc_tr = self.tgt[self.mis_c]
                    self.misc_pred = self.predicted[self.mis_c]
                    self.flg =1
                else:  
                    self.misc_im = torch.cat((self.inputs[self.mis_c],self.misc_im))
                    self.misc_tr = torch.cat((self.tgt[self.mis_c],self.misc_tr))
                    self.misc_pred = torch.cat((self.predicted[self.mis_c],self.misc_pred))            
    
        self.test_loss /= len(self.testloader.dataset)
        self.accuracy = 100. * self.correct / len(self.testloader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            self.test_loss, self.correct, len(self.testloader.dataset),self.accuracy
            ))
        self.test_loss_graph.append(self.test_loss)
        self.test_accuracy_graph.append(self.accuracy)
        return self.test_loss_graph, self.test_accuracy_graph, self.misc_im,self.misc_tr,self.misc_pred
    def run_samples(self, epochs, train_loader, test_loader):
        self.epochs = epochs
        self.trainloader = train_loader
        self.testloader = test_loader
        self.train_loss_graph = []
        self.train_accuracy_graph = []
        self.test_loss_graph = []
        self.test_accuracy_graph = []
        for self.epoch in range(self.start_epoch, self.start_epoch+self.epochs):
            self.train_loss_graph, self.train_accuracy_graph, self.accuracy = self.train(self.epoch, self.trainloader, self.device, self.train_loss_graph, self.train_accuracy_graph)
            self.scheduler.step(self.accuracy)
            self.test_loss_graph, self.test_accuracy_graph, self.misc_im,self.misc_tr,self.misc_pred  = self.test(self.epoch, self.testloader, self.device, self.test_loss_graph, self.test_accuracy_graph)
        return self.train_loss_graph, self.train_accuracy_graph, self.test_loss_graph, self.test_accuracy_graph, self.misc_im,self.misc_tr,self.misc_pred
