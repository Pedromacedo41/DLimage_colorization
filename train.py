import torch
import torch.nn as nn

import sys
sys.path.append('./models_pytorch')

from models_pytorch.colorization_deploy_v1 import colorization_deploy_v1

def main():
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


if __name__ == '__main__': 
    main()

