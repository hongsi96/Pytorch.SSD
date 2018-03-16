import os
import opts
import argparse
import random
import numpy as np 
from buildbox import MakeBox
from numpy.random import RandomState
import cv2

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from evaluation import eval_voc_detection
from model import SSD
import data

#setting
parser = argparse.ArgumentParser(description='Single Shot Multi Detector')
opts.setting(parser)
opt = parser.parse_args()

#random_seed
PRNG = RandomState(opt.seed)

# model
model = SSD(opt.n_classes)
cfg = model.config
decoder = MakeBox(cfg)
model.load_state_dict(torch.load(opt.weight_path))
model.cuda()
cudnn.benchmark = True

#dataload
dataset = data.loader_test(cfg, opt.data_path ,PRNG)
print('size of dataset:',len(dataset))

def test():
    print('testing....')
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = [], [], [], [], []
    for i in range(len(dataset)):
        img, loc, label = dataset[i]
        
        gt_bboxes.append(loc)
        gt_labels.append(label)

        input = Variable(img.unsqueeze(0), volatile=True)
        input = input.cuda()

        xloc, xconf = model(input)
        xloc = xloc.data.cpu().numpy()[0]
        xconf = xconf.data.cpu().numpy()[0]

        boxes, labels, scores = decoder.decode(xloc, xconf, nms_thresh=0.5, conf_thresh=0.01)

        pred_bboxes.append(boxes)
        pred_labels.append(labels)
        pred_scores.append(scores)

    print(eval_voc_detection(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_thresh=0.5, use_07_metric=True))

        
        


if __name__ == '__main__':
    test()