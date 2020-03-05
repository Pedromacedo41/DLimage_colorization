import torch
import torch.nn as nn
import numpy as np 

# to transform image to Lab color scale
from skimage import io, color
from models.colorization_deploy_v1 import colorization_deploy_v1
from utils.color_quantization import NNEncode
from utils.weights import PriorFactor
import os


use_gpu = torch.cuda.is_available()
def gpu(tensor, gpu=use_gpu):
    if gpu:
        return tensor.cuda()
    else:
        return tensor


sigma = 5
nb_neighboors = 10
ENC_DIR = './utils/'

# encoder_decoder ab to Q space
nnenc = NNEncode(nb_neighboors,sigma,km_filepath=os.path.join(ENC_DIR,'pts_in_hull.npy'))

# weights for balanced loss
priors = PriorFactor(1, gamma= 0.5, priorFile=os.path.join(ENC_DIR,'prior_probs.npy'))

def loss(input, img_ab):
    '''
    img_ab = np.ones(shape= (1,2,224,224))
    imput = torch.ones([1,313,224,224], dtype = torch.float64)
    gpu(imput)
    ''''

    d2 = torch.tensor(nnenc.encode_points_mtx_nd(img_ab), dtype = torch.float64)
    # dimension 1 x 224 x 224
    gpu(d2)
    weights = priors.compute(imput)
 
    z2 = torch.sum(-imput.log_().mul_(d2), dim=1)
    z2.mul_(weights)

    return z2.sum()

def main():

    # parameter of weights in Z= H_gt^-1(Y)
    

    # balancing between balanced classes weighted loss and average loss
    lamb = 0.5

    batch_size = 50
    lr = 1e-4
    nb_epochs = 1000

    loss_epoch = []

    net = colorization_deploy_v1(T=0.38)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)

    result = net(torch.ones([1,1,224,224]))
    print(loss(result.detach().numpy(),result.detach().numpy()))

    '''
    for e in range(nb_epochs):
  
        loss = 0  

        # define some dataloader
        # result= net(input)
        
        # compute loss
        # loss = f(result, ground_truth)
          
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss+= loss
                        
        loss_epoch.append(loss)
     '''




if __name__ == '__main__': 
    main()

