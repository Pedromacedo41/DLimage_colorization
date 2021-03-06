import os
import secrets
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
    parser.add_argument('--images', default=None, required=True, type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--focal', action='store_true')

    args = parser.parse_args()
    return args


use_gpu = torch.cuda.is_available()
def gpu(tensor, gpu=use_gpu, device=None):
    if gpu:
        return tensor.cuda()
    else:
        return tensor


# sigma = 5
# nb_neighboors = 10
# ENC_DIR = './utils/'

# encoder_decoder ab to Q space
# nnenc = NNEncode(nb_neighboors,sigma,km_filepath=os.path.join(ENC_DIR,'pts_in_hull.npy'))

# weights for balanced loss
# priors = PriorFactor(1, gamma= 0.5, priorFile=os.path.join(ENC_DIR,'prior_probs.npy'))


# def loss_fn(imput, img_ab, device='cpu:0'):
#     #d2 = gpu(torch.tensor(nnenc.encode_points_mtx_nd(img_ab.numpy()), dtype= torch.float32))
#     d2 = torch.tensor(nnenc.encode_points_mtx_nd(img_ab.numpy()), dtype= torch.float32).to(device)
 
#     z2 = -imput.log()
#     del d2

#     z2 = torch.sum(z2, dim=1)

#     weights = priors.compute(imput).to(device)
#     z2 = z2.mul(weights)

#     return z2.sum()

def logist_mask(x):
    return torch.sigmoid(40*(x-.5))

def focal_loss(_input, input_ab, reduction='mean'):
    output = f.mse_loss(_input, input_ab, reduction='none')

    # Max pixel loss
    max_ = torch.max(output)

    # Compute relative loss
    aux = output.detach()
    aux = aux / max_

    # Mask
    aux = logist_mask(aux)
    output = output.mul(aux)

    if reduction=='mean':
        return output.sum()/aux.sum()
    elif reduction == 'sum':
        return output.sum()
    elif reduction == 'none':
        return output
    else:
        raise 'Invalid mode'


def plot(im, interp=False):
    f = plt.figure(figsize=(5,10), frameon=True)
    plt.imshow(im, interpolation=None if interp else 'none')

# return list of index of 10 best images colorization and 10 worst images colorizations
def test(args):

    folder = f'output-{secrets.token_hex(4)}'
    print(folder)

    os.makedirs(folder)
    os.makedirs(f'{folder}/best')
    os.makedirs(f'{folder}/worst')

    if args.focal:
        loss_fn = focal_loss
    else:
        loss_fn = f.mse_loss

    with torch.no_grad():
        batch_size = 224
        losses = []

        dataset = ImageDataset(args.images)
        dataloader = DataLoader(dataset, batch_size, False, num_workers=16)

        model = colorization_deploy_v1(T=0.38)

        if args.focal:
            model.load_state_dict(torch.load('model_l2_focal.pt'))
        else:
            model.load_state_dict(torch.load('model_l2.pt'))

        model.eval()
        model = nn.DataParallel(model)
        model = gpu(model)

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

        # for i in range(10):
        #     im, real = dataset.get_img(i)
        #     im = gpu(im)
        #     im = model.predict_rgb(im)
        #     io.imsave(f'scenes/rand/{i}_pred.png', im)
        #     io.imsave(f'scenes/rand/{i}_real.png', real)

        sorted_losses = sorted(losses,key=itemgetter(0))

        list_best = [a[1] for a in sorted_losses[0:50]] 
        list_worst = [a[1] for a in sorted_losses[-51:-1]] 


        for i in list_best:
            im, real = dataset.get_img(i)
            im = gpu(im)
            im = model.module.predict_rgb(im)
            io.imsave(f'{folder}/best/{i}_pred.png', im)
            io.imsave(f'{folder}/best/{i}_real.png', real)

        for i in list_worst:
            im, real = dataset.get_img(i)
            im = gpu(im)
            im = model.module.predict_rgb(im)
            io.imsave(f'{folder}/worst/{i}_pred.png', im)
            io.imsave(f'{folder}/worst/{i}_real.png', real)


def train(args, n_epochs=100, load_model=False):
    if args.focal:
        loss_fn = focal_loss
    else:
        loss_fn = f.mse_loss

    batch_size = 224
    lr = 1e-4

    dataset = ImageDataset(args.images)
    dataloader = DataLoader(dataset, batch_size, True, num_workers=16, pin_memory=False)

    model = colorization_deploy_v1(T=0.38)

    if load_model:
        if args.focal:
            model.load_state_dict(torch.load('model_l2_focal.pt'))
        else:
            model.load_state_dict(torch.load('model_l2.pt'))
    
    model = nn.DataParallel(model)
    model = gpu(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    n_data = len(dataloader.dataset)

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
        if args.focal:
            try:
                os.replace('model_l2_focal.pt', 'model_l2_focal_prev.pt')
            except:
                pass
            torch.save(model.module.state_dict(), 'model_l2_focal.pt')
        else:
            try:
                os.replace('model_l2.pt', 'model_l2_prev.pt')
            except:
                pass
            torch.save(model.module.state_dict(), 'model_l2.pt')


def main():
    args = parse_args()
    if(args.train):
        train(args, load_model = not args.new)
    else:
        test(args)


if __name__ == '__main__': 
    main()

