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

def demo():
    print('practice ssd (demo) ....')
    MEAN = [123.68, 116.779, 103.939] 
    DRAW = draw()
    
    i = PRNG.choice(len(dataset))
    j=0
    if not os.path.isdir('demoimages'):
            os.mkdir('demoimages')

    for _ in range(100):
        img, _, _ = dataset[i]
        input = Variable(img.unsqueeze(0), volatile=True)
        input = input.cuda()
        xloc, xconf = model(input)
        j=j+1
        imgs = input.data.cpu().numpy().transpose(0, 2, 3, 1)
        xloc = xloc.data.cpu().numpy()
        xconf = xconf.data.cpu().numpy()
        
        for img, loc, conf in zip(imgs, xloc, xconf):
            img = ((img[:, :, ::-1] + MEAN)).astype('uint8')
            boxes, labels, scores = decoder.decode(loc, conf, conf_thresh=0.5)

            img = DRAW.draw_bbox(img, boxes, labels, True)

        if j<50:
            demopath1 = os.path.join('demoimages','test07_demo'+str(j)+'.jpg')
            cv2.imwrite(demopath1,img)

        i = PRNG.choice(len(dataset))
            
class draw(object):
    def __init__(self):
        setting = opts.class_list()

        classes = setting.CLASSES

        self.id_to_label = setting.id_to_label
        self.label_to_id = setting.label_to_id

        colors = {}
        for label in classes:
            id = self.label_to_id[label]
            color = self._to_color(id, len(classes))
            colors[id] = color
            colors[label] = color
        self.colors =colors

    def _to_color(self, indx, n_classes):
        base = int(np.ceil(pow(n_classes, 1./3)))
        base2 = base * base
        b = 2 - indx / base2
        r = 2 - (indx % base2) / base
        g = 2 - (indx % base2) % base
        return (r * 127, g * 127, b * 127)

    def draw_bbox(self, img, bboxes, labels, relative=False):
        if len(labels) == 0:
            return img
        img = img.copy()
        h, w = img.shape[:2]

        if relative:
            bboxes = bboxes * [w, h, w, h]

        bboxes = bboxes.astype(np.int)
        labels = labels.astype(np.int)

        for bbox, label in zip(bboxes, labels):
            left, top, right, bot = bbox
            color = self.colors[label]
            label = self.id_to_label[label]
            cv2.rectangle(img, (left, top), (right, bot), color, 2)
            cv2.putText(img, label, (left+1, top-5), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)

        return img


if __name__ == '__main__':
    demo()