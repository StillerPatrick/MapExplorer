from torch.utils.data import Dataset

import numpy as np
import torch
import os 
import cv2
from tqdm import tqdm


class Mapdataset(Dataset):

    def __init__(self, basedir, gpu=True):
        """
        Loads the data, resizes the data and provides x and y
        """
        y_path = os.path.join(basedir,"NoBackground")
        x_path = os.path.join(basedir,"WithBackground")

        print(x_path)

        self.xs = []
        self.ys = []

        for file in tqdm((os.listdir(x_path)[:2000]),desc="Load inputs"): 
            image = cv2.imread(os.path.join(x_path,file),0)
            image = cv2.resize(image,(400,75))
            image = np.expand_dims(image,0)
            self.xs.append(image)

        for file in tqdm((os.listdir(y_path)[:2000]),desc="Load Labels"): 
            image = cv2.imread(os.path.join(y_path,file),0)
            image = cv2.resize(image,(400,75))
            _,image = cv2.threshold(image,180,255,cv2.THRESH_BINARY)
            image = np.expand_dims(image,0)
            self.ys.append(image)

        if gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        print(len(self.xs))
        print(len(self.ys))


    def __getitem__(self, index):
        """
        Get a specific item
        """
        return torch.Tensor(self.xs[index]).float().cuda(), torch.Tensor(self.ys[index]).float().cuda()

    def __len__(self):
        """
        Get length of the training data set
        """
        return len(self.xs)
    
 