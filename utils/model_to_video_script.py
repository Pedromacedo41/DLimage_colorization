import argparse
import cv2
from skimage import color

import numpy as np


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


def main(args):
    
    cap = cv2.VideoCapture("y2mate.mp4")
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
        
        fra2 = color.rgb2lab(frame)
        fra2[:,:,1:3]= 0
        #print(fra2)
        #break
        fra3 = cv2.Lab2RGB(fra2)
        cv2.imshow('frame',fra3)
        out.write(fra3)

        '''
        if(i==0):
            fra2 = color.rgb2lab(frame)
            fra3 = color.lab2rgb(fra2)
            print(frame[0,0,:])
            print("kkl")
            print(fra3[0,0,:]/fra3[0,0,:].min() )
            break
        '''

        i+=1
    
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    args = parse_args()
    main(args)