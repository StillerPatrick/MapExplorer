import argparse
import torch 
#import horovod.torch as hvd 
import numpy as np 
from dataset import Mapdataset
from models.unet import UNet
from loss.loss import SSIM
from tensorboardX import SummaryWriter
import tools as tools 
from tqdm import tqdm

# define command line parameter 
parser = argparse.ArgumentParser(description="Define the parameter of the training process")

# add arguments for the argument parser

parser.add_argument("--epochs", action="store", type=int)
parser.add_argument("--batchsize",action="store",type=int)
parser.add_argument("--basedir",action="store",type=str)
parser.add_argument("--tbpath", action="store",type=str)
parser.add_argument("--gpu", action="store", type=int)
parser.add_argument("--shuffle", action="store", type=int)
parser.add_argument("--identifier",action="store", type=str)



args = parser.parse_args()

trainDataset = Mapdataset(args.basedir, args.gpu)
trainLoader = torch.utils.data.DataLoader(trainDataset,args.batchsize,args.shuffle,num_workers=0)
writer = SummaryWriter(args.tbpath)

if args.gpu:
    model = UNet(1,1).cuda()
else:
    model = UNet(1,1)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

ssim = SSIM(device="cuda:0" if args.gpu else "cpu:0")

# training loop 
for epoch in range(args.epochs):
    for train_x, train_y in tqdm(trainLoader, desc='Train epoch %d' % epoch):
            optimizer.zero_grad()
            prediction = model(train_x)
            loss = -ssim(prediction,train_y)
            loss.backward()
            optimizer.step()
    
    tools.saveimage(prediction, train_y,train_x,epoch)
    print("Loss at Epoch",epoch,":",loss.item())
        



