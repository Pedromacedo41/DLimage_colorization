import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 
import argparse

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

def logist_mask(relative_loss):
    """
    relative_loss = pixel_loss / max_loss (<= 1)
    """
    return 1 / (1 + 20 * torch.exp(-15 * relative_loss))

def focal_loss(input, img_ab):
    gpu(imput)
    d2 = torch.tensor(nnenc.encode_points_mtx_nd(img_ab), dtype= torch.float32)
    # dimension 1 x 224 x 224
    gpu(d2)
    weights = priors.compute(imput)
 
    z2 = torch.sum(-imput.log().mul(d2), dim=1)
    gpu(z2)
    z2 = z2.mul(weights)

    # Max pixel loss
    max_ = torch.max(z2)

    # Copy z2 to a new tensor
    Rel_loss = z2.clone()
    Rel_loss.requires_grad_(False)

    Rel_loss = Rel_loss / max_

    # Mask
    mask_ = logist_mask(Rel_loss)
    z2 = z2.mul(mask_.data)

    return z2.sum()

def train(args, n_epochs=100, load_model=False):
    batch_size = 224
    lr = 1e-4

    dataset = ImageDataset(args.images)
    dataloader = DataLoader(dataset, batch_size, True, num_workers=16, pin_memory=True)

    model = colorization_deploy_v1(T=0.38)

    if load_model:
        model.load_state_dict(torch.load('model_l2.pt'))
    
    model = nn.DataParallel(model)
    model = gpu(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    n_data = len(dataloader.dataset)

    loss_fn = nn.MSELoss()

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
            os.replace('model_l2.pt', 'model_l2_prev.pt')
        except:
            pass
        torch.save(model.module.state_dict(), 'model_l2.pt')


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
    train(args, load_model=True)

