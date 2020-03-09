import torch.nn as nn

import torch.nn.functional as F
import torch

import numpy as np
import os
import argparse


class colorization_deploy_v1(nn.Module):
    def __init__(self, T=0.38, decoding_layer= False):
        super(colorization_deploy_v1, self).__init__()
        self.T= T
        self.deconding_layer = decoding_layer

        self.conv1_1 = nn.Conv2d(1,64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64,64, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256,256, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilation= 2, padding=2)
        self.conv5_2 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilation= 2, padding=2)
        self.conv5_3 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilation= 2, padding=2)

        self.conv6_1 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilation= 2, padding=2)
        self.conv6_2 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilation= 2, padding=2)
        self.conv6_3 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilation= 2, padding=2)

        self.conv7_1 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.conv7_3 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.ConvTranspose2d(512,256, kernel_size=4, stride=2, padding=1)
        self.conv8_2 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        self.conv8_3 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)

        self.conv8_313 = nn.Conv2d(256,313, kernel_size=1, stride=1)
        self.conv_ab = nn.Conv2d(313,2, kernel_size=1, stride=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)


        self.lin = nn.Sequential(
            self.conv1_1,
            nn.ReLU(),
            self.conv1_2,
            nn.ReLU(),
            self.bn1,

            self.conv2_1, 
            nn.ReLU(),
            self.conv2_2, 
            nn.ReLU(),
            self.bn2,

            self.conv3_1, 
            nn.ReLU(),  
            self.conv3_2, 
            nn.ReLU(),  
            self.conv3_3,  
            nn.ReLU(), 
            self.bn3,

            self.conv4_1, 
            nn.ReLU(),  
            self.conv4_2, 
            nn.ReLU(),  
            self.conv4_3, 
            nn.ReLU(), 
            self.bn4, 

            self.conv5_1, 
            nn.ReLU(), 
            self.conv5_2, 
            nn.ReLU(),   
            self.conv5_3,  
            nn.ReLU(),   
            self.bn5,  

            self.conv6_1,  
            nn.ReLU(),     
            self.conv6_2,  
            nn.ReLU(),     
            self.conv6_3, 
            nn.ReLU(),  
            self.bn6,    

            self.conv7_1,
            nn.ReLU(),      
            self.conv7_2,  
            nn.ReLU(),    
            self.conv7_3,  
            nn.ReLU(),  
            self.bn7, 

            self.conv8_1,
            nn.ReLU(),
            self.conv8_2,
            nn.ReLU(),
            self.conv8_3,
            nn.ReLU(),

            self.conv8_313,

            nn.Upsample(scale_factor=4, mode='nearest')

        )

       

    def forward(self, input):
        out = F.softmax(self.lin(input), dim=1)
        if(self.deconding_layer==True):
            return self.conv_ab(out)
        else:
            return out

 

 
if __name__ == '__main__': 
    net = colorization_deploy_v1()
    t = net(torch.ones([1,1,224,224]))
    # print(t)
    print(t.shape)
