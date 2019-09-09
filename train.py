import argparse
import torch 
import horovod.torch as hvd 
import numpy as np 
from dataset import Mapdataset

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

dataset = Mapdataset(args.basedir,args.numTrainingSamples, args.numValidationSamples, args.numTestSamples, args.gpu)





# create dataset 



# define trainloop


    # iterate over all elements of the dataset 