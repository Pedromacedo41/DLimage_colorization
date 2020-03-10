import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 
import argparse
from matplotlib import pyplot as plt
from operator import itemgetter

# to transform image to Lab color scale
from skimage import io, color
from models.colorization_deploy_v1 import colorization_deploy_v1
from utils.color_quantization import NNEncode
from utils.weights import PriorFactor

from utils.data import ImageDataset

sigma = 5
nb_neighboors = 10
ENC_DIR = './utils/'

# encoder_decoder ab to Q space
nnenc = NNEncode(nb_neighboors,sigma,km_filepath=os.path.join(ENC_DIR,'pts_in_hull.npy'))

# weights for balanced loss
priors = PriorFactor(1, gamma= 0.5, priorFile=os.path.join(ENC_DIR,'prior_probs.npy'))

def plot(im, interp=False):
    f = plt.figure(figsize=(5,10), frameon=True)
    plt.imshow(im, interpolation=None if interp else 'none')

use_gpu = torch.cuda.is_available()
def gpu(tensor, gpu=use_gpu, device=None):
    if gpu:
        return tensor.cuda()
    else:
        return tensor

def loss_fn(imput, img_ab, device='cpu:0'):
    #d2 = gpu(torch.tensor(nnenc.encode_points_mtx_nd(img_ab.numpy()), dtype= torch.float32))
    d2 = torch.tensor(nnenc.encode_points_mtx_nd(img_ab.numpy()), dtype= torch.float32).to(device)
 
    z2 = -imput.log()
    del d2

    z2 = torch.sum(z2, dim=1)

    weights = priors.compute(imput).to(device)
    z2 = z2.mul(weights)

    return z2.sum()

def test(model, dataloader):
    batch_size = 1
    losses = []

    model = gpu(model)
    
    n_data = len(dataloader.dataset)

    processed = 0
    i=0
    for inputs, inputs_ab, classes in dataloader:
        inputs = gpu(inputs)
        inputs_ab = inputs_ab.detach()
        outputs = model(inputs)
    
        loss = loss_fn(outputs, inputs_ab, 'cuda:0')
        losses.append((loss, i))

        processed += inputs.shape[0]
        print(f'Processed {processed} out of {n_data}: {100*processed/n_data} %')
        i+=1

    
    sorted_losses = sorted(losses,key=itemgetter(0))

    list_best = [a[0] for a in sorted_losses[0:10]] 
    list_worse = [a[0] for a in sorted_losses[-11:-1]] 

    return list_best, list_worse


def main():
    dataset = ImageDataset("Images")
    dataloader = DataLoader(dataset, 1, False, num_workers=16)
    model = torch.load('./../base.pt')
    F= test(model,dataloader)
    print(F)

if __name__ == '__main__': 
    # torch.autograd.set_detect_anomaly(True)
    main()