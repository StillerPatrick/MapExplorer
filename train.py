import argparse
import torch 
import horovod.torch as hvd 
import numpy as np 
from dataset import Mapdataset
from models.unet import UNet
from loss.loss import SSIM
# define command line parameter 
parser = argparse.ArgumentParser(description="Define the parameter of the training process")

# add arguments for the argument parser

parser.add_argument("--epochs", action="store", type=int)
parser.add_argument("--batchsize",action="store",type=int)
parser.add_argument("--basedir",action="store",type=str)
parser.add_argument("--tbpath", action="store",type=str)
parser.add_argument("--numTrainingSamples", action="store", type=int)
parser.add_argument("--numValidationSamples", action="store", type=int)
parser.add_argument("--numTestSamples", action="store", type=int)
parser.add_argument("--gpu", action="store", type=int)
parser.add_argument("--shuffle", action="store", type=int)
parser.add_argument("--identifier",action="store", type=int)

args = parser.parse_args()

trainDataset = Mapdataset(args.basedir, args.gpu)
trainLoader = torch.utils.data.DataLoader(trainDataset,args.batch_size,args.shuffle,num_workers=2)

model = UNet(3,1)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

ssim = SSIM()

# training loop 
for epoch in range(args.epochs):
    for train_x, train_y in trainLoader:
            optimizer.zero_grad()
            prediction = model(train_x)
            loss = -ssim(train_y, prediction)
            loss.backward()
            optimizer.step()
            if epoch % int(args.epoch / 10):
                print("Loss at Epoch",epoch+1,":",loss.item())
        



