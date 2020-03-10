import argparse
import cv2
from skimage import color
import sys
import torch
from torchvision import transforms
from skimage import color
from PIL import Image
from color_quantization import NNEncode
from matplotlib import pyplot as plt

import numpy as np
sys.path.append('./../models')

from colorization_deploy_v1 import colorization_deploy_v1


sigma = 5
nb_neighboors = 10

nnenc = NNEncode(nb_neighboors,sigma,km_filepath='pts_in_hull.npy')

'''
Script for making a colorful video from a legacy black and white one
'''

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default="none", type=str)
    parser.add_argument('--input_file', default="none", type=str)
    parser.add_argument('--output_file', default="none", type=str)

    args = parser.parse_args()
    return args


def return_model():
    #model = colorization_deploy_v1(T=0.38)
    #model.load_state_dict(torch.load("converted.h5"))
    #model.eval()
    model = torch.load("./../../base.pt")
    return model

def plot(im, interp=False):
    f = plt.figure(figsize=(5,10), frameon=True)
    plt.imshow(im, interpolation=None if interp else 'none')

def image():
    img_rgb = Image.open("sun.jpg")
    model = return_model()
    plot(model.predict(img_rgb))

def main(args):
    
    model = return_model()
    cap = cv2.VideoCapture("test.mp4")
    print(cap)

    # Define the codec and create VideoWriter object
    
    ret, frame = cap.read() 
    height,width,layer=frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter('video.avi',fourcc,15.0,(width,height))
    i = 0
    while(1):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            #cv2.destroyAllWindows()
            break
        

        #fra2[:,:,1:3]= 0
    
        #out.write(fra3)

        #cv2.imshow("", frame)
        if(i==0):
            #cv2.imshow("", frame)
            rgb = color.rgb2lab(frame)
            #rgb = transforms.Resize((256,256), Image.BICUBIC)(rgb)
            rgb = np.array(rgb)
            Lab = color.rgb2lab(rgb).astype(np.float32).transpose(2,0,1)
            l = Lab[0,:,:][np.newaxis, np.newaxis,...]
            
            tr = model(torch.as_tensor(l))
            tr = nnenc.decode_points_mtx_nd(tr.detach().numpy())
            
            #break
    

        i+=1
    
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    args = parse_args()
    main(args)
    image()