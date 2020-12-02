import torch
import torchvision
import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms, utils


class CrowdDataset(Dataset):

    def __init__(self, root, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with examples and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.examples = pd.read_csv(os.path.join(root, csv_file))
        self.root = root
        # self.transform = transform

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        grid_dim = 7
        img_dim = 448
        len_of_grid = img_dim / grid_dim
        img_path = os.path.join(self.root, self.examples.iloc[idx, 0])
        image = Image.open(img_path)
        annotations_path = os.path.join(self.root, self.examples.iloc[idx, 1])
        annotation_dict = json.load(open(annotations_path, "r"))
        annotations = annotation_dict["boxes"]
        if len(annotations) is 0: 
          boxes =  np.array([[0,0,1,1]])
          labels = np.zeros(1)
        else:
          boxes =  np.asarray(annotations)
          labels = np.ones(len(boxes))
       
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.455, 0.422, 0.409],
                                 std=[0.193, 0.181, 0.178])
        ])
        
        x = t(image) 
        targets = {} 
        targets['boxes'] = torch.from_numpy(boxes).float() 
        targets['labels'] = torch.from_numpy(labels).type(torch.int64)
        return x, targets