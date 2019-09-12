from torch.utils.data import Dataset

import numpy as np
import torch
import os 
import cv2
from tqdm import tqdm


class Mapdataset(Dataset):

    def __init__(self, basedir, gpu=True, length=2000):
        """
        Loads the data, resizes the data and provides x and y
        """
        y_path = os.path.join(basedir,"NoBackground")
        x_path = os.path.join(basedir,"WithBackground")

        print(x_path)

        self.images_x = [os.path.join(x_path,file) for file in os.listdir(x_path)[:length]]
        self.images_y = [os.path.join(y_path,file) for file in os.listdir(y_path)[:length]]

        if gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    @staticmethod
    def convert_image(filename):
        image = cv2.imread(filename,0)
        image = cv2.resize(image,(400,75))
        image = np.expand_dims(image,0)
        image = image /255
        return self.dtype(image)

    def __getitem__(self, index):
        """
        Get a specific item
        """
        y = self.convert_image(self.images_y[index])
        x = self.convert_image(self.images_x[index])
        return x,y

    def __len__(self):
        """
        Get length of the training data set
        """
        return len(self.images_y)

    @staticmethod
    def get_image(path):
        image = cv2.imread(path,0)
        image = cv2.resize(image,(400,75))
        image = np.expand_dims(image,0)
        image = image / 255
        return torch.Tensor(image)
    
 