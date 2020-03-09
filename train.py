import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np 

# to transform image to Lab color scale
from skimage import io, color
from models.colorization_deploy_v1 import colorization_deploy_v1
from utils.color_quantization import NNEncode
from utils.weights import PriorFactor

from utils.data import ImageDataset


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

def loss_fn(imput, img_ab):
    d2 = gpu(torch.tensor(nnenc.encode_points_mtx_nd(img_ab.numpy()), dtype= torch.float32))
    weights = priors.compute(imput)
 
    z2 = -imput.log().mul(d2)
    z2 = torch.sum(z2, dim=1)
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

def train(n_epochs=4):
    batch_size = 32
    lr = 1e-4

    dataset = ImageDataset('images')
    dataloader = DataLoader(dataset, batch_size, True, num_workers=4)

    model = colorization_deploy_v1(T=0.38)
    # pp = 0
    # for parameter in model.parameters():
    #     nn = 1
    #     for s in parameter.size():
    #         nn = nn*s
    #     pp += nn
    # print(pp)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for e in range(n_epochs):
        running_loss = 0
        for inputs, inputs_ab, classes in dataloader:
            inputs_ab = inputs_ab.detach()
            outputs = model(inputs)
            print(inputs.shape, outputs.shape, inputs_ab.shape)
            optimizer.zero_grad()
            loss = loss_fn(outputs, inputs_ab)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        print(f'Epoch: {e}\nMean loss: {running_loss/len(dataloader.dataset)}\n')


def main():

    net = colorization_deploy_v1(T=0.38)
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)

    img_ab = torch.ones((3,2,224,224))
    output = net(torch.ones([3,1,224,224]))

    gpu(output)

    print(loss_fn(output, img_ab))


if __name__ == '__main__': 
    torch.autograd.set_detect_anomaly(True)
    train()

