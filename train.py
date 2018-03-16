import os
import opts
import argparse
import random


import numpy as np 
from numpy.random import RandomState
import torch
import cv2
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pdb
from model import SSD
from loss import MultiBoxLoss
import data


#setting
parser = argparse.ArgumentParser(description='Single Shot Multi Detector')
opts.setting(parser)
opt = parser.parse_args()
init_lr = opt.lr
print('')
print('Setting:', opt)
print('')


#random_seed
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
PRNG = RandomState(opt.seed)


# model

model = SSD(opt.n_classes)
cfg = model.config
model.init_parameters(opt.pretrainedvgg)
criterion = MultiBoxLoss()
model.cuda()
criterion.cuda()
cudnn.benchmark = True
#print(cfg)
#print('')


#dataload
dataset = data.loader(cfg, opt.augmentation, opt.data_path ,PRNG)
print('size of dataset:', len(dataset))

# optimizer
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


def train():   
    model.train()
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads, pin_memory=True)

    iteration = opt.iter_start
    os.mkdir(opt.w_file_name)
    print('training....')
    while iteration<opt.iter_finish:
        for input, loc, label in dataloader:
            lr_update(iteration, optimizer)
            input, loc, label = Variable(input), Variable(loc), Variable(label)
            input, loc, label = input.cuda(), loc.cuda(), label.cuda()
            xloc, xconf = model(input)
            loc_loss, conf_loss = criterion(xloc, xconf, loc, label)
            loss = loc_loss + conf_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % opt.iter_save == 0:
                print('iteration: {}, Loss:{}'.format(iteration, loss.data[0]))
                torch.save(model.state_dict(), opt.w_file_name+'/{}_{}.pth'.format(cfg.get('name', 'SSD'), iteration))
            
            iteration += 1   

def lr_update(iteration, optimizer):
    if iteration in opt.lr_shedule:   
        opt.lr = opt.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr


if __name__ == '__main__':
    train()