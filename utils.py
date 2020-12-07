
import PIL
import torch
import json
import os
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#Label
voc_labels = ('person')

label_map = {k: v+1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
#Inverse mapping
rev_label_map = {v: k for k, v in label_map.items()}

#Colormap for bounding box
CLASSES = 1


def save_label_map(output_path):
    '''
        Save label_map to output file JSON
    '''
    with open(os.path.join(output_path, "label_map.json"), "w") as j:
        json.dump(label_map, j)


def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)
        
        Out: Intersection each of boxes1 with respect to each of boxes2, 
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy =  torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                        boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))
    
    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy , min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  #(n1, n2)

def find_IoU(boxes1, boxes2):
    '''
        Find IoU between every boxes set of boxes 
        boxes1: a tensor of dimensions (n1, 4) (left, top, right , bottom)
        boxes2: a tensor of dimensions (n2, 4)
        
        Out: IoU each of boxes1 with respect to each of boxes2, a tensor of 
             dimensions (n1, n2)
        
        Formula: 
        (box1 ∩ box2) / (box1 u box2) = (box1 ∩ box2) / (area(box1) + area(box2) - (box1 ∩ box2 ))
    '''
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter) #(n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  #(n1, n2)
    union = (area_boxes1 + area_boxes2 - inter)
    return inter / union
#==========================END CACULATE IoU====================================

def combine(batch):
    '''
        Combine these tensors of different sizes in batch.
        batch: an iterable of N sets from __getitem__()
    '''
    images = []
    boxes = []
    labels = []
    # difficulties = []
    
    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        # difficulties.append(b[3])
        
    images = torch.stack(images, dim= 0)
    # return images, boxes, labels, difficulties
    return images, boxes, labels


#=====================BEGIN CONVERT BBOXES=======================================
def xy_to_cxcy(bboxes):
    '''
        Convert bboxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
        bboxes: Bounding boxes, a tensor of dimensions (n_object, 4)
        
        Out: bboxes in center coordinate
    '''
    return torch.cat([(bboxes[:, 2:] + bboxes[:, :2]) / 2,
                      bboxes[:, 2:] - bboxes[:, :2]], 1)
        
def cxcy_to_xy(bboxes):
    '''
        Convert bboxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    '''
    return torch.cat([bboxes[:, :2] - (bboxes[:, 2:] / 2),
                      bboxes[:, :2] + (bboxes[:, 2:] / 2)], 1)

def encode_bboxes(bboxes,  default_boxes):
    '''
        Encode bboxes correspoding default boxes (center form)
        
        Out: Encodeed bboxes to 4 offset, a tensor of dimensions (n_defaultboxes, 4)
    '''
    return torch.cat([(bboxes[:, :2] - default_boxes[:, :2]) / (default_boxes[:, 2:] / 10),
                      torch.log(bboxes[:, 2:] / default_boxes[:, 2:]) *5],1)

def decode_bboxes(offsets, default_boxes):
    '''
        Decode offsets
    '''
    return torch.cat([offsets[:, :2] * default_boxes[:, 2:] / 10 + default_boxes[:, :2], 
                      torch.exp(offsets[:, 2:] / 5) * default_boxes[:, 2:]], 1)

def adjust_lr(optimizer, scale):
    '''
        Scale learning rate by a specified factor
        optimizer: optimizer
        scale: factor to multiply learning rate with.
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

def save_checkpoint(epoch, model, optimizer):
    '''
        Save model checkpoint
    '''
    state = {'epoch': epoch, "model": model, "optimizer": optimizer}
    filename = "model.pth.tar"
    torch.save(state, filename)
    
def decimate(tensor, m):
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())
    return tensor
