import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Layer_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Layer_block, self).__init__()
        self.padding = (kernel_size - 1) // 2     ###auto same padding###
        self.conv = nn.Sequential()
        self.conv.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                stride=1, padding=self.padding))
        self.conv.add_module('bn1', nn.BatchNorm2d(out_channels))
        self.conv.add_module('maxpool', nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=self.padding))
        self.resident = nn.Sequential()
        self.resident.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
        self.resident.add_module('bn1', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.conv(x) + self.resident(x)
    
    

class ResNet(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 10 ) -> object:
        super(ResNet, self).__init__()
        self.block1= Layer_block(input_channels, out_channels =15, kernel_size = 7,stride =  2)
        self.block2 = Layer_block(15, 30, kernel_size = 3, stride = 2)
        self.block3 = Layer_block(30, 60, kernel_size = 3, stride = 2)
        self.block4 = Layer_block(60, 120, kernel_size = 3, stride = 4)
        self.fc1 = nn.Conv2d(120,7680, kernel_size = 8, stride = 1, padding = 0)
        self.fc2 = nn.Linear(7680, 512) 
        self.fc3 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu( self.block1(x)) 
        x = F.relu( self.block2(x))
        x = F.relu( self.block3(x))
        x = F.relu( self.block4(x))
        x = F.relu( self.fc1(x) )
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        x = F.relu( x)
        x = self.fc3(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
summary(model, (3, 256, 256))
    
