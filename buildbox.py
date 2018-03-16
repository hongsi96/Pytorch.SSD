import numpy as np
import torch

import itertools
from numbers import Number
import l2norm
import pdb

class MakeBox(object):
    def __init__(self, cfg):
        self.pos_thresh = cfg.get('pos_thresh', 0.5)
        self.neg_thresh = cfg.get('neg_thresh', 0.5)
        self.prior_variance = cfg.get('prior_variance', [0.1, 0.1, 0.2, 0.2])

        steps = cfg.get('steps', None)
        grids = cfg['grids']
        sizes = cfg['sizes']
        aspect_ratios = cfg['aspect_ratios']

        anchor_boxes = []
        for k in range(len(grids)):
            w, h = (grids[k], grids[k]) 
            #pdb.set_trace()
            step_w, step_h = (steps[k], steps[k]) 

            for u, v in itertools.product(range(h), range(w)):  
                cx = (v + 0.5) * step_w
                cy = (u + 0.5) * step_h

                s = np.sqrt(sizes[k] * sizes[k+1])
                anchor_boxes.append([cx, cy, s, s])

                s = sizes[k]
                for ar in aspect_ratios[k]:
                    anchor_boxes.append([cx, cy, s * np.sqrt(ar), s * np.sqrt(1. / ar)])

        self.anchor_boxes = np.array(anchor_boxes)      
        self.anchor_boxes_ = np.hstack([                
            self.anchor_boxes[:, :2] - self.anchor_boxes[:, 2:] / 2,
            self.anchor_boxes[:, :2] + self.anchor_boxes[:, 2:] / 2])   


    def encode(self, boxes, labels):
        
        if len(boxes) == 0:
            return (
                torch.FloatTensor(np.zeros(self.anchor_boxes.shape, dtype=np.float32)),
                torch.LongTensor(np.zeros(self.anchor_boxes.shape[0], dtype=np.int)))

        #pdb.set_trace()
        iou = l2norm.batch_iou(self.anchor_boxes_, boxes) 
        idx = iou.argmax(axis=1)
        #print(idx)
        #print('gg')

        iouc = iou.copy()
        for _ in range(len(boxes)):
            i, j = np.unravel_index(iouc.argmax(), iouc.shape)
            if iouc[i, j] < 0.1:
                continue
            iouc[i, :] = 0
            iouc[:, j] = 0

            idx[i] = j 
            iou[i, j] = 1.
        iou = iou.max(axis=1)
        #print(iou)
        #print('ll')
        #print(len(boxes))
        boxes = boxes[idx]
        #print(len(boxes))
        loc = np.hstack([
                ((boxes[:, :2] + boxes[:, 2:]) / 2. - self.anchor_boxes[:, :2]) / self.anchor_boxes[:, 2:],
                np.log((boxes[:, 2:] - boxes[:, :2]) / self.anchor_boxes[:, 2:]),
                ]) / self.prior_variance
        
        labels = labels[idx]
        labels = 1 + labels
        labels[iou < self.neg_thresh] = 0
        labels[(self.neg_thresh <= iou) & (iou < self.pos_thresh)] = -1   
        #print(len(loc[1]))
        #print(len(labels))

        return torch.FloatTensor(loc.astype(np.float32)), torch.LongTensor(labels.astype(np.int))

    
    def decode(self, loc, conf, nms_thresh=0.5, conf_thresh=0.5):
        loc = loc * self.prior_variance
        boxes = np.hstack([
                    loc[:, :2] * self.anchor_boxes[:, 2:] + self.anchor_boxes[:, :2],
                    np.exp(loc[:, 2:]) * self.anchor_boxes[:, 2:]])
        boxes[:, :2], boxes[:, 2:] = (boxes[:, :2] - boxes[:, 2:] / 2., 
                                      boxes[:, :2] + boxes[:, 2:] / 2.)
        boxes = np.clip(boxes, 0, 1)

        conf = np.exp(conf)
        conf /= conf.sum(axis=-1, keepdims=True)
        scores = conf[:, 1:]

        chosen = np.zeros(len(scores), dtype=bool)
        for i in range(scores.shape[1]):
            keep = nms(boxes, scores[:, i], nms_thresh, conf_thresh)
            scores[:, i] *= keep
            chosen |= keep

        chosen &= (-scores.max(axis=1)).argsort().argsort() < 200
        return boxes[chosen], scores.argmax(axis=1)[chosen], scores.max(axis=1)[chosen]

def nms(boxes, scores, nms_thresh=0.45, conf_thresh=0, topk=400, topk_after=50):
    Keep = np.zeros(len(scores), dtype=bool)
    idx =  (scores >= conf_thresh) & ((-scores).argsort().argsort() < topk)
    if idx.sum() == 0:
        return Keep

    boxes = boxes[idx]
    scores = scores[idx]

    iou = l2norm.batch_iou(boxes, boxes)
    keep = np.zeros(len(scores), dtype=bool)
    keep[scores.argmax()] = True
    for i in scores.argsort()[::-1]:
        if (iou[i, keep] < nms_thresh).all():
            keep[i] = True

    Keep[idx] = keep
    return Keep