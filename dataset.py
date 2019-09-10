from torch.utils.data import Dataset

import numpy as np
import torch
import os 
import cv2
from tqdm import tqdm


class Mapdataset(Dataset):

    def __init__(self, basedir, gpu=True):
        y_path = os.path.join(basedir,"NoBackground")
        x_path = os.path.join(basedir,"WithBackground")

        print(x_path)

        self.xs = []
        self.ys = []

        for file in tqdm((os.listdir(x_path)[:2]),desc="Load inputs"): 
            image = cv2.imread(os.path.join(x_path,file))
            image = cv2.resize(image,(800,150))
            image = np.rollaxis(image,2,0)
            self.xs.append(image)

        for file in tqdm((os.listdir(y_path)[:2]),desc="Load Labels"): 
            image = cv2.imread(os.path.join(y_path,file))
            image = cv2.resize(image,(800,150))
            image = np.rollaxis(image,2,0)
            self.ys.append(image)

        if gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        print(len(self.xs))
        print(len(self.ys))


    def __getitem__(self, index):
        return self.dtype(self.xs[index]), self.dtype(self.ys[index])

    def __len__(self):
        return len(self.xs)

    @staticmethod
    def get_image(path):
        image = cv2.imread(path)
        image = cv2.resize(image,(800,150))
        image = np.rollaxis(image,2,0)
        return torch.Tensor(image)
    
 