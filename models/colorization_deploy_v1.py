import torch.nn as nn

import torch.nn.functional as F
from skimage import color
import torch

import numpy as np
import os
import argparse
import scipy.ndimage.interpolation as sni


class colorization_deploy_v1(nn.Module):
    def __init__(self, T=0.38, ab_mode= False):
        super(colorization_deploy_v1, self).__init__()
        self.T= T

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

        # self.conv8_313 = nn.Conv2d(256,313, kernel_size=1, stride=1)
        self.conv_ab = nn.Conv2d(256, 2, kernel_size=1, stride=1)

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
            self.conv_ab,

            nn.Upsample(scale_factor=4, mode='bicubic')
        )

    def forward(self, input):
        return self.lin(input)

    def lab2rgb(self, input_l, input_ab):
        lab = np.concatenate
    
    def predict_rgb(self, input_l):
        pred_ab = self.forward(input_l)*256 - 128
        input_l *= 100
        pred_ab = pred_ab.squeeze(0).cpu()
        input_l = input_l.squeeze(0).cpu()
        lab = np.concatenate((input_l, pred_ab), axis=0)
        lab = lab.transpose((1,2,0))
        rgb = color.lab2rgb(lab)
        return rgb
        #img_lab = color.rgb2lab(input) # convert image to lab color space
        #img_l = img_lab[:,:,0] # pull out L channel
        ## (H_orig,W_orig) = input.size[:2] # original image size
        #mean_img_l = torch.as_tensor(img_l-50, dtype=torch.float32)
        #mean_img_l.unsqueeze_(0).unsqueeze_(0)
        #pred_ab = self.forward(mean_img_l).squeeze(0)
        #d = pred_ab.detach().numpy().transpose((1, 2, 0))
        ## upsample to match size of original image L
        #ab_dec_us = sni.zoom(d, (4, 4, 1))
        ## concatenate with original image L
        #img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
        #img_rgb_out = np.clip(color.lab2rgb(256*img_lab_out-128),
        #                      0, 1)  # convert back to rgb
        #return img_rgb_out
        

    def fill_weights(self):
        model = torch.load("colorization_release_v1.caffemodel.pt")

        self.conv8_1.weight = nn.Parameter(model["conv8_1.weight"])
        self.conv8_1.bias = nn.Parameter(model["conv8_1.bias"])

        self.conv8_2.weight = nn.Parameter(model["conv8_2.weight"])
        self.conv8_2.bias = nn.Parameter(model["conv8_2.bias"])

        self.conv8_3.weight = nn.Parameter(model["conv8_3.weight"])
        self.conv8_3.bias = nn.Parameter(model["conv8_3.bias"])

        self.conv7_1.weight = nn.Parameter(model["conv7_1.weight"])
        self.conv7_1.bias = nn.Parameter(model["conv7_1.bias"])

        self.conv7_2.weight = nn.Parameter(model["conv7_2.weight"])
        self.conv7_2.bias = nn.Parameter(model["conv7_2.bias"])

        self.conv7_3.weight = nn.Parameter(model["conv7_3.weight"])
        self.conv7_3.bias = nn.Parameter(model["conv7_3.bias"])

        self.bn7.weight = nn.Parameter(model["conv7_3norm.weight"])
        self.bn7.bias = nn.Parameter(model["conv7_3norm.bias"])
        #6
        self.conv6_1.weight = nn.Parameter(model["conv6_1.weight"])
        self.conv6_1.bias = nn.Parameter(model["conv6_1.bias"])

        self.conv6_2.weight = nn.Parameter(model["conv6_2.weight"])
        self.conv6_2.bias = nn.Parameter(model["conv6_2.bias"])

        self.conv6_3.weight = nn.Parameter(model["conv6_3.weight"])
        self.conv6_3.bias = nn.Parameter(model["conv6_3.bias"])

        self.bn6.weight = nn.Parameter(model["conv6_3norm.weight"])
        self.bn6.bias = nn.Parameter(model["conv6_3norm.bias"])
        #5
        self.conv5_1.weight = nn.Parameter(model["conv5_1.weight"])
        self.conv5_1.bias = nn.Parameter(model["conv5_1.bias"])

        self.conv5_2.weight = nn.Parameter(model["conv5_2.weight"])
        self.conv5_2.bias = nn.Parameter(model["conv5_2.bias"])

        self.conv5_3.weight = nn.Parameter(model["conv5_3.weight"])
        self.conv5_3.bias = nn.Parameter(model["conv5_3.bias"])

        self.bn5.weight = nn.Parameter(model["conv5_3norm.weight"])
        self.bn5.bias = nn.Parameter(model["conv5_3norm.bias"])
        #4
        self.conv4_1.weight = nn.Parameter(model["conv4_1.weight"])
        self.conv4_1.bias = nn.Parameter(model["conv4_1.bias"])

        self.conv4_2.weight = nn.Parameter(model["conv4_2.weight"])
        self.conv4_2.bias = nn.Parameter(model["conv4_2.bias"])

        self.conv4_3.weight = nn.Parameter(model["conv4_3.weight"])
        self.conv4_3.bias = nn.Parameter(model["conv4_3.bias"])

        self.bn4.weight = nn.Parameter(model["conv4_3norm.weight"])
        self.bn4.bias = nn.Parameter(model["conv4_3norm.bias"])

        #3
        self.conv3_1.weight = nn.Parameter(model["conv3_1.weight"])
        self.conv3_1.bias = nn.Parameter(model["conv3_1.bias"])

        self.conv3_2.weight = nn.Parameter(model["conv3_2.weight"])
        self.conv3_2.bias = nn.Parameter(model["conv3_2.bias"])

        self.conv3_3.weight = nn.Parameter(model["conv3_3.weight"])
        self.conv3_3.bias = nn.Parameter(model["conv3_3.bias"])

        self.bn3.weight = nn.Parameter(model["conv3_3norm.weight"])
        self.bn3.bias = nn.Parameter(model["conv3_3norm.bias"])
        
        #2
        self.conv2_1.weight = nn.Parameter(model["conv2_1.weight"])
        self.conv2_1.bias = nn.Parameter(model["conv2_1.bias"])

        self.conv2_2.weight = nn.Parameter(model["conv2_2.weight"])
        self.conv2_2.bias = nn.Parameter(model["conv2_2.bias"])


        self.bn2.weight = nn.Parameter(model["conv2_2norm.weight"])
        self.bn2.bias = nn.Parameter(model["conv2_2norm.bias"])

        #1
        self.conv1_1.weight = nn.Parameter(model["bw_conv1_1.weight"])
        self.conv1_1.bias = nn.Parameter(model["bw_conv1_1.bias"])

        self.conv1_2.weight = nn.Parameter(model["conv1_2.weight"])
        self.conv1_2.bias = nn.Parameter(model["conv1_2.bias"])


        self.bn1.weight = nn.Parameter(model["conv1_2norm.weight"])
        self.bn1.bias = nn.Parameter(model["conv1_2norm.bias"])

        #print(model["class8_ab.weights"].shape)
        self.conv_ab.weight = nn.Parameter(torch.rand([2, 256, 1, 1]))
        self.conv_ab.bias = nn.Parameter(torch.rand([2]))


if __name__ == '__main__': 
    net = colorization_deploy_v1(decoding_layer=True)
    t = net(torch.ones([1,1,224,224]))
    net.fill_weights()
    torch.save(net, "base.pt")
    # print(t)
    #print(t.shape)
