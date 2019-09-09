from torch.utils.data import Dataset

import numpy as np
import torch
import os 
import cv2

class Mapdataset(Dataset):

    def __init__(self, basedir, gpu=True):
        y_path = os.path.join(basedir,"NoBackground")
        x_path = os.path.join(basedir,"WithBackground")

        self.xs = []
        self.ys = []
        for file in x_path: 
            self.xs.append(cv2.imread(file))

        for file in y_path: 
            self.ys.append(cv2.imread(file,0))


    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def __len__(self):
        return len(self.xs)
    
 