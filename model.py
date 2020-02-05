import torch.nn as nn 
from functools import reduce
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Generator(nn.Module):
    '''
    Generator model 

    output images with given vectors.
    '''
    def __init__(self, feature_dim=128, hidden_dim = [256, 512, 1024], out_shape=(1, 28, 28), leaky_slope=0.2):
        super(Generator, self).__init__()
        
        out_dim = reduce(lambda x, y: x * y, out_shape)
        dims = [feature_dim] + hidden_dim

        self.feature_dim = feature_dim
        self.out_shape = out_shape
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]), 
                nn.LeakyReLU(leaky_slope)
            ) for i in range(len(dims) - 1)
        ] + 
        [
            nn.Sequential(
                nn.Linear(dims[-1], out_dim),
                nn.Tanh()
            )
        ]
        )

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x).view(-1, *self.out_shape)
    
    def _train(self, optimizer, criterion, discriminator, fake_data, device):
        B = fake_data.size(0)

        optimizer.zero_grad()
        loss = criterion(discriminator(fake_data), torch.ones((B, 1)).to(device))

        loss.backward()
        optimizer.step()

        return loss.cpu().detach()

class Discriminator(nn.Module):
    '''
    Discriminator model 

    output images with given vectors.
    '''
    def __init__(self, in_dim=28**2, hidden_dim = [1024, 512, 256], leaky_slope=0.2, drop_out=0.3):
        super(Discriminator, self).__init__()

        dims = [in_dim] + hidden_dim

        self.in_dim = in_dim
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]), 
                nn.LeakyReLU(leaky_slope),
                nn.Dropout(drop_out)
            ) for i in range(len(dims) - 1)
        ] + 
        [
            nn.Sequential(
                nn.Linear(dims[-1], 1),
                nn.Sigmoid()
            )
        ]
        )

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        return reduce(lambda x, l: l(x), self.layers, x)

    def _train(self, optimizer, criterion, real_data, fake_data, device):
        B = real_data.size(0)

        optimizer.zero_grad()

        loss_real = criterion(self.forward(real_data), torch.ones((B, 1)).to(device))
        loss_fake = criterion(self.forward(fake_data), torch.zeros((B, 1)).to(device))

        loss = loss_real + loss_fake
        loss.backward()
        optimizer.step()

        return loss.cpu().detach()

