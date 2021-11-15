import torch.nn as nn
import torch
import torch.nn.functional as F

################ Define AutoEncoder Using MLP ################
"""
class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()

        self.ch_num = 1
        self.class_num = 10

        ################ Define Network Layers ################
        # define convolution layers
        self.conv1 = nn.Conv2d(self.ch_num, 5, 5, 1, 2,bias=True)
        self.conv2 = nn.Conv2d(5, 6, 2, 2, 0,bias=True)
        self.conv3 = nn.Conv2d(6, 16, 5, 1, 0,bias=True)
        self.conv4 = nn.Conv2d(16, 16, 2, 2, 0, bias=True)
        self.conv5 = nn.Conv2d(16, 120, 5, 1, 0, bias=True)
        self.fc1 = nn.Linear(120,84, bias=True)
        self.fc2 = nn.Linear(84,self.class_num,bias=True)



    def forward(self, x):
        #################### Network  ####################
        #print(x.size())
        # 28->14
        x = self.conv1(x)
        x = torch.tanh(x)
        #print(x.size())

        # 14->7
        x = self.conv2(x)
        x = torch.tanh(x)
        #print(x.size())

        x = self.conv3(x)
        x = torch.tanh(x)
        #print(x.size())

        x = self.conv4(x)
        x = torch.tanh(x)
        #print(x.size())

        x = self.conv5(x)
        x = torch.tanh(x)
        #print(x.size())

        x = x.view(-1,120)
        #print(x.size())

        x = self.fc1(x)
        x = torch.tanh(x)
        #print(x.size())

        x = self.fc2(x)
        #print(x.size())

        # we should apply softmax
        x = F.log_softmax(dim=x)
        return x

"""


################ Define AutoEncoder Using MLP ################
class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()

        self.ch_num = 1
        self.class_num = 2
        self.filter_base_num = 64



        ################ Define Network Layers ################
        # define convolution layers
        self.conv1 = nn.Conv2d(self.ch_num, self.filter_base_num, 4, 2, 1,bias=True)
        self.conv2 = nn.Conv2d(self.filter_base_num, 2*self.filter_base_num, 4, 2, 1,bias=True)
        self.conv3 = nn.Conv2d(2*self.filter_base_num, 10*self.class_num, 7, 1, 0,bias=True)
        self.fc1 = nn.Linear(10*self.class_num, self.class_num, bias=True)
        # define batch normalization layers
        self.en_bn1 = nn.BatchNorm2d(self.filter_base_num)
        self.en_bn2 = nn.BatchNorm2d(2*self.filter_base_num)




    def forward(self, x):
        #################### Network  ####################
        #print(x.size())
        # 28->14
        x = self.conv1(x)
        x = self.en_bn1(x)
        x = F.leaky_relu(x)
        #print(x.size())

        # 14->7
        x = self.conv2(x)
        x = self.en_bn2(x)
        x = F.leaky_relu(x)
        #print(x.size())

        # 7->1 (here we do not impose any constraints on code part)
        x = self.conv3(x)
        #print(x.size())

        x = x.view(-1, 10*self.class_num)
        x = self.fc1(x)

        # we should apply softmax
        # x = F.log_softmax(x)
        return x




