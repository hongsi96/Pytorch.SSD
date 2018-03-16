import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

import numpy as np
import os
import itertools
from collections import OrderedDict
from l2norm import L2Norm
from vgg import VGG16




class SSD(nn.Module):

    config = {'name': 'SSD300-VGG16', 'image_size': 300,
             'grids': (38, 19, 10, 5, 3, 1), 
             'aspect_ratios': ((1/2.,  1,  2), (1/3.,  1/2.,  1,  2,  3), (1/3.,  1/2.,  1,  2,  3), (1/3.,  1/2.,  1,  2,  3), (1/2.,  1,  2), (1/2.,  1,  2)),
             'steps': [s / 300. for s in [8, 16, 32, 64, 100, 300]],
             'sizes': [s / 300. for s in [30, 60, 111, 162, 213, 264, 315]]}                                                 
    
    def __init__(self, n_classes):
        super(SSD, self).__init__()
        self.n_classes = n_classes

        self.Base = VGG16()
        self.Extra = nn.Sequential(OrderedDict([('extra1_1', nn.Conv2d(1024, 256, 1)), ('extra1_2', nn.Conv2d(256, 512, 3, padding=1, stride=2)), ('extra2_1', nn.Conv2d(512, 128, 1)),
            ('extra2_2', nn.Conv2d(128, 256, 3, padding=1, stride=2)), ('extra3_1', nn.Conv2d(256, 128, 1)),
            ('extra3_2', nn.Conv2d(128, 256, 3)), ('extra4_1', nn.Conv2d(256, 128, 1)), ('extra4_2', nn.Conv2d(128, 256, 3))]))
        self.pred_layers = ['conv4_3', 'conv7', 'extra1_2', 'extra2_2', 'extra3_2', 'extra4_2']
        n_channels = [512, 1024, 512, 256, 256, 256]

        self.L2Norm = nn.ModuleList([L2Norm(512, 20)])
        self.norm_layers = ['conv4_3']  

        self.Loc = nn.ModuleList([])
        self.Conf = nn.ModuleList([])
        for i, ar in enumerate(self.config['aspect_ratios']):
            n = len(ar) + 1
            self.Loc.append(nn.Conv2d(n_channels[i], n * 4, 3, padding=1))
            self.Conf.append(nn.Conv2d(n_channels[i], n * (self.n_classes + 1), 3, padding=1))

    def forward(self, x):
        xs = []
        for name, m in itertools.chain(self.Base._modules.items(), 
                                       self.Extra._modules.items()):
            if isinstance(m, nn.Conv2d):
                x = F.relu(m(x), inplace=True)
            else:
                x = m(x)

            if name in self.pred_layers:
                if name in self.norm_layers:
                    i = self.norm_layers.index(name)
                    xs.append(self.L2Norm[i](x))
                else:
                    xs.append(x)

        return self._prediction(xs)

    def _prediction(self, xs):
        locs = []
        confs = []
        for i, x in enumerate(xs):
            loc = self.Loc[i](x)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)

            conf = self.Conf[i](x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0), -1, self.n_classes + 1)
            confs.append(conf)
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)


    def init_parameters(self, pretrainedvgg=None):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.zero_()
        self.apply(weights_init)

        if pretrainedvgg is not None and os.path.isfile(pretrainedvgg):
            self.Base.load_pretrained(pretrainedvgg)
            print('pretrained vgg loaded!')
        else:
            print('No pretrained vgg file!')

