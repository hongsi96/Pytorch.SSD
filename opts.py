import argparse
import torch
import torch.nn as nn
import numpy as np

# Setup
def setting(parser):
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--data_path',type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--iter_start', type=int,  default=1)
    parser.add_argument('--iter_save', type=int,  default=50)
    parser.add_argument('--iter_finish', type=int, default=10000)
    parser.add_argument('--w_file_name', type=str, default='weight_gggg')
    parser.add_argument('--lr_shedule', type=int, default=(4000, 8000))
    parser.add_argument('--pretrainedvgg', default='')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--augmentation', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=20)


class class_list(object):
    N_CLASSES = 20
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')
    label_to_id = dict(map(reversed, enumerate(CLASSES))) 
    id_to_label = dict(enumerate(CLASSES)) 


