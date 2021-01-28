## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input size 96 x 96
        # output size = (W-F)/S + 1 = (96-4)/1 +1 = 93 
        # after one pool layer, this becomes (32, 46, 46)
        self.conv1 = nn.Conv2d(1, 32, 4) 
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # input (32, 46, 46)
        # output size = (W-F)/S+1 = (46 - 3)/1+1 = 44
        # after one pool layer, this becomes (64, 22, 22)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # input (64, 22, 22)
        # output size = (W-F)/S+1 = (22 - 2)/1+1 = 21
        # after one pool layer, this becomes (128, 10, 10)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        # input (128, 10, 10)
        # output size = (W-F)/S+1 = (10 - 1)/1+1 = 10
        # after one pool layer, this becomes (256, 5, 5)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_drop = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(1000, 68*2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        # prep for linear layer 
        x = x.view(x.size(0), -1)
        
        # three linear layers with dropout in between 
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
