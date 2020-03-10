import os
import torch
import torch.nn as nn
import torch.nn.functional as f
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


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--images', default='images', type=str)

    args = parser.parse_args()
    return args


use_gpu = torch.cuda.is_available()
def gpu(tensor, gpu=use_gpu, device=None):
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


def loss_fn(imput, img_ab, device='cpu:0'):
    #d2 = gpu(torch.tensor(nnenc.encode_points_mtx_nd(img_ab.numpy()), dtype= torch.float32))
    d2 = torch.tensor(nnenc.encode_points_mtx_nd(img_ab.numpy()), dtype= torch.float32).to(device)
 
    z2 = -imput.log()
    del d2

    z2 = torch.sum(z2, dim=1)

    weights = priors.compute(imput).to(device)
    z2 = z2.mul(weights)

    return z2.sum()

def logist_mask(x):
    return torch.sigmoid(40*(x-.5))

def focal_loss(_input, input_ab):
    output = f.mse_loss(_input, input_ab, reduction='none')

    # Max pixel loss
    max_ = torch.max(output)

    # Compute relative loss
    aux = output.detach()
    aux = aux / max_

    # Mask
    aux = logist_mask(aux)
    output = output.mul(aux)

    return output.sum()/aux.sum()

def plot(im, interp=False):
    f = plt.figure(figsize=(5,10), frameon=True)
    plt.imshow(im, interpolation=None if interp else 'none')

# return list of index of 10 best images colorization and 10 worst images colorizations
def test(args):
    with torch.no_grad():
        batch_size = 224
        losses = []

        dataset = ImageDataset(args.images)
        dataloader = DataLoader(dataset, batch_size, False, num_workers=16)

        model = colorization_deploy_v1(T=0.38)
        model.load_state_dict(torch.load('model_l2.pt'))
        model.eval()
        #model = nn.DataParallel(model)
        model = gpu(model)

        im = gpu(torch.from_numpy(dataloader.dataset[0][0]).unsqueeze(0))
        im = model.predict_rgb(im)
        io.imsave('image.png', im)
        return

        n_data = len(dataloader.dataset)

        processed = 0
        i=0
        for inputs, inputs_ab, classes in dataloader:

            inputs = gpu(inputs)
            inputs_ab = gpu(inputs_ab.detach())
            outputs = model(inputs)

            bs = inputs.shape[0]
        
            loss = f.mse_loss(outputs, inputs_ab, reduction='none').view(bs, -1).mean(1)
            for l in loss:
                i+=1
                losses.append((l.item(), i))

            processed += inputs.shape[0]
            print(f'Processed {processed} out of {n_data}: {100*processed/n_data} %')

        sorted_losses = sorted(losses,key=itemgetter(0))

        list_best = [a[1] for a in sorted_losses[0:10]] 
        list_worse = [a[1] for a in sorted_losses[-11:-1]] 

        #print(list_best) 
        #print(list_worse)

        return list_best, list_worse

def train(args, n_epochs=100, load_model=False):
    batch_size = 224
    lr = 1e-4

    dataset = ImageDataset(args.images)
    dataloader = DataLoader(dataset, batch_size, True, num_workers=16, pin_memory=False)

    model = colorization_deploy_v1(T=0.38)

    if load_model:
        model.load_state_dict(torch.load('model_l2_focal.pt'))
    
    model = nn.DataParallel(model)
    model = gpu(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    n_data = len(dataloader.dataset)

    loss_fn = focal_loss

    for e in range(n_epochs):
        running_loss = 0
        processed = 0
        for inputs, inputs_ab, classes in dataloader:
            inputs = gpu(inputs)
            inputs_ab = gpu(inputs_ab)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_fn(outputs, inputs_ab)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            processed += inputs.shape[0]
            #print(f'Loss: {loss.item()}')
            #print(f'Processed {processed} out of {n_data}: {100*processed/n_data} %')
        print(f'Epoch: {e}\nMean loss: {running_loss}\n')
        try:
            os.replace('model_l2_focal.pt', 'model_l2_focal_prev.pt')
        except:
            pass
        torch.save(model.module.state_dict(), 'model_l2_focal.pt')


def main():

    net = colorization_deploy_v1(T=0.38)
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)

    img_ab = torch.ones((3,2,224,224))
    output = net(torch.ones([3,1,224,224]))

    gpu(output)

    print(loss_fn(output, img_ab))


if __name__ == '__main__': 
    #torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    #train(args, load_model=True)
    best, worst = test(args)

