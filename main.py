#!/usr/bin/env python3
import matplotlib as mpl
mpl.use('Agg') # for SSH service
from argparse import ArgumentParser
import numpy as np
import torch
from model import *
from data import *
from train import *


# parser args
parser = ArgumentParser('Vanilla GAN')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--D-epochs', type=int, default=1)
parser.add_argument('--G-epochs', type=int, default=1)
# parser.add_argument('--hidden-layers', type=int, default=3)
parser.add_argument('--lr', type=int, default=0.0001)
parser.add_argument('--feature-dim', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=128) # 128
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--test', action='store_true')

if __name__ == '__main__':
    # hyper-params
    G_HIDDEN_LAYERS = [256, 512, 1024]
    D_HIDDEN_LAYERS = [1024, 512, 256]

    # parse the args
    args = parser.parse_args()

    # whether to use cuda
    cuda = torch.cuda.is_available() and args.cuda

    # dataset
    train_dataset = get_dataset('mnist')
    test_dataset = get_dataset('mnist', train=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = 28 ** 2

    # cuda 
    device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
    
    G = Generator(args.feature_dim, G_HIDDEN_LAYERS).to(device)
    D = Discriminator(28**2, D_HIDDEN_LAYERS).to(device)

    if not args.test:
        train(G, D, args, train_dataloader, device)
    