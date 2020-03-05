import torch
import torch.nn as nn

# to transform image to Lab color scale
from skimage import io, color
from models_pytorch.colorization_deploy_v1 import colorization_deploy_v1
from resources.Color_quantization import *
import sklearn.neighbors as nn


def main():

    sigma = 5
    nnenc = NNEncode(nn,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

    # parameter of weights in Z= H_gt^-1(Y)
    

    # balancing between balanced classes weighted loss and average loss
    lamb = 0.5

    batch_size = 50
    lr = 1e-4
    nb_epochs = 1000

    loss_epoch = []

    net = colorization_deploy_v1()
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)

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



def loss(input, img_rgb):
    



if __name__ == '__main__': 
    main()

