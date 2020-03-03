import torch.nn as nn

import torch.nn.functional as F

import numpy as np
import os
import argparse


class colorization_deploy_v1(nn.Module):
    def __init__(self, T=0.38):
        super(generator,self).__init__()
        self.T= T

        self.conv1_1 = nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64,64, kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256,256, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512,512, kernel_size=3, stride=2, padding=1)

        self.conv5_1 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilatation=2, padding=1)
        self.conv5_2 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilatation= 2, padding=1)
        self.conv5_3 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilatation= 2, padding=1)

        self.conv6_1 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilatation= 2, padding=1)
        self.conv6_2 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilatation= 2, padding=1)
        self.conv6_3 = nn.Conv2d(512,512, kernel_size=3, stride=1, dilatation= 2, padding=1)

        self.conv7_1 = nn.Conv2d(512,256, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        self.conv7_3 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(256,128, kernel_size=3, stride=0.5, padding=1)
        self.conv8_2 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.conv8_3 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)

        self.conv8_313 = nn.Conv2d(128,313, kernel_size=1, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(256)

        self.softm = nn.softmax(313)
       

    def forward(self, input):

        # First block
        out = input
        out = self.bn1(F.relu(self.conv1_2(F.relu(self.conv1_1(out)))))
        out = self.bn2(F.relu(self.conv2_2(F.relu(self.conv2_1(out)))))

        out = self.bn3(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(out)))))))
        out = self.bn4(F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(out)))))))
        out = self.bn5(F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(out)))))))
        out = self.bn6(F.relu(self.conv6_3(F.relu(self.conv6_2(F.relu(self.conv6_1(out)))))))
        out = self.bn7(F.relu(self.conv7_3(F.relu(self.conv7_2(F.relu(self.conv7_1(out)))))))
        out = self.conv_313(F.relu(self.conv8_3(F.relu(self.conv8_2(F.relu(self.conv8_1(out)))))))
     
        # annealing mean computation missing
        return self.sofm(out/self.T)