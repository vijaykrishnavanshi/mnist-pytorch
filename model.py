import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=3)
        init.xavier_uniform(self.conv1.weight, gain=numpy.sqrt(2.0))
        init.constant(self.conv1.bias, 0.1)
        self.conv2 = nn.Conv2d(100, 25, kernel_size=3)
        init.xavier_uniform(self.conv1.weight, gain=numpy.sqrt(2.0))
        init.constant(self.conv1.bias, 0.1)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3025, 100)
        self.fc2 = nn.Linear(100,10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        #x = self.conv2_drop(x)
        #x = F.max_pool2d(x, 2)
        x = F.relu(x)    
        print(x.size())
        x = x.view(-1, 3025)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    