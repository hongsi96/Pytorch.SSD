import numpy as np
import torch
import torch.nn as nn
import itertools
from numbers import Number

import pdb



def batch_iou(a, b):  

    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    inter = np.clip(rb - lt, 0, None)

    area_i = np.prod(inter, axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)

    area_u = area_a[:, np.newaxis] + area_b - area_i
    return area_i / np.clip(area_u, 1e-7, None)  


    

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale=20):
        super(L2Norm,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x /= (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out