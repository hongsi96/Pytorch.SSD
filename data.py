import os
from opts import class_list
import argparse
import numpy as np 
import torch
import cv2
from buildbox import MakeBox
from functions import *
import xml.etree.ElementTree as ET 
import pdb
import torch.utils.data



def loader(cfg, augmentation, root,PRNG):
    encoder = MakeBox(cfg)
    if augmentation:
        print('yes')
        transform = Compose([
                [ColorJitter(prob=0.5)], 
                BoxesToCoords(),
                Expand((1, 4), prob=0.5),
                ObjectRandomCrop(),
                HorizontalFlip(),
                Resize(300),
                CoordsToBoxes(),
                [SubtractMean(mean=MEAN)],
                [RGB2BGR()],
                [ToTensor()],
                ], PRNG, mode=None, fillval=MEAN)
    else:
        print('No')
        transform = Compose([
                BoxesToCoords(),
                Resize(300),
                CoordsToBoxes(),
                [SubtractMean(mean=MEAN)],
                [RGB2BGR()],
                [ToTensor()]])
    target_transform = encoder.encode

    dataset = Detection(root=root, image_set=[('2012', 'trainval')], keep_difficult=True, transform=transform, target_transform=target_transform)
    return dataset



def loader_test(cfg, root,PRNG):
    transform = Compose([
            BoxesToCoords(),
            Resize(300),
            CoordsToBoxes(),
            [SubtractMean(mean=MEAN)],
            [RGB2BGR()],
            [ToTensor()]])

    dataset = Detection(root=root, image_set=[('2007', 'test')], transform=transform, target_transform=None)
    return dataset


MEAN = [123.68, 116.779, 103.939] 

class ParseAnnotation(object):
    def __init__(self, keep_difficult=True):
        self.keep_difficult = keep_difficult

        voc = class_list()
        self.label_to_id = voc.label_to_id
        self.classes = voc.CLASSES

    def __call__(self, target):
        tree = ET.parse(target).getroot()

        bboxes = []
        labels = []
        for obj in tree.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue

            label = obj.find('name').text.lower().strip()
            if label not in self.classes:
                continue
            label = self.label_to_id[label]

            bndbox = obj.find('bndbox')
            bbox = [int(bndbox.find(_).text) - 1 for _ in ['xmin', 'ymin', 'xmax', 'ymax']]

            bboxes.append(bbox)
            labels.append(label)

        return np.array(bboxes), np.array(labels)



class Detection(torch.utils.data.Dataset):
    def __init__(self, root, image_set, keep_difficult=False, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        #self.augmentation = augmentation
        self.target_transform = target_transform

        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')

        self.parse_annotation = ParseAnnotation(keep_difficult=keep_difficult)

        self.ids = []
        for year, split in image_set:
            basepath = os.path.join(self.root, 'VOC' + str(year))
            path = os.path.join(basepath, 'ImageSets', 'Main')
            for file in os.listdir(path):
                if not file.endswith('_' + split + '.txt'):
                    continue
                with open(os.path.join(path, file)) as f:
                    for line in f:
                        self.ids.append((basepath, line.strip()[:-3]))
        #print('lamda')
        #pdb.set_trace()
        self.ids = sorted(list(set(self.ids)), key=lambda _:_[0]+_[1])  # deterministic 
        #pdb.set_trace()

    def __getitem__(self, index):
        img_id = self.ids[index]

        img = cv2.imread(self._imgpath % img_id)[:, :, ::-1]
        bboxes, labels = self.parse_annotation(self._annopath % img_id)

        if self.transform is not None:
            img, bboxes = self.transform(img, bboxes)

        bboxes, labels = self.filter(img, bboxes, labels)
        if self.target_transform is not None:
            bboxes, labels = self.target_transform(bboxes, labels)
        return img, bboxes, labels


    def __len__(self):
        return len(self.ids)

    def filter(self, img, boxes, labels):
        shape = img.shape
        if len(shape) == 2:
            h, w = shape
        else:   
            if shape[0] > shape[2]:  
                h, w = img.shape[:2]
            else:                     
                h, w = img.shape[1:]

        boxes_ = []
        labels_ = []
        for box, label in zip(boxes, labels):
            if min(box[2] - box[0], box[3] - box[1]) <= 0:
                continue
            if np.max(boxes) < 1 and np.sqrt((box[2] - box[0]) * w * (box[3] - box[1]) * h) < 8:
                continue
            boxes_.append(box)
            labels_.append(label)
        return np.array(boxes_), np.array(labels_)




