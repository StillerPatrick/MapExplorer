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

        self.xs = []
        self.ys = []

        for file in os.listdir(x_path)[:length]:
            try:
                image = cv2.imread(os.path.join(x_path,file),0)
                image = cv2.resize(image,(400,75))
                image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
                image = np.expand_dims(image,0)
                image = image /255
                self.xs.append(image)
            except Exception as e: print(e,image)
                
                
        for file in os.listdir(y_path)[:length]: 
            try:
                image = cv2.imread(os.path.join(y_path,file),0)
                image = cv2.resize(image,(400,75))
                image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
                image = np.expand_dims(image,0)
                image = image /255
                self.ys.append(image)
            except Exception as e: print(e,image)

        if gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

<<<<<<< HEAD
=======
    def convert_image(self, filename):
        image = cv2.imread(filename,0)
        _,image = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
        image = cv2.resize(image,(400,75))
        image = np.expand_dims(image,0)
        image = image // 255
        return self.dtype(image)


>>>>>>> 7c8bb2ea359cc302c286a199c69734075ea421e0
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

    def get_image(path):
        image = cv2.imread(path,0)
        _,image = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
        image = cv2.resize(image,(400,75))
        image = np.expand_dims(image,0)
        image = image // 255
        return torch.Tensor(image)
    
    
 