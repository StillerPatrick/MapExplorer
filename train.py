import argparse
import torch 
#import horovod.torch as hvd 
import numpy as np 
from dataset import Mapdataset
from models.unet import UNet
from models.tiramisu import FCDenseNet103
from loss.loss import SSIM
from tensorboardX import SummaryWriter
import tools as tools 
from tqdm import tqdm
import os 

# define command line parameter 
parser = argparse.ArgumentParser(description="Define the parameter of the training process")

# add arguments for the argument parser

parser.add_argument("--epochs", action="store", type=int)
parser.add_argument("--batchsize",action="store",type=int)
parser.add_argument("--basedirtrain",action="store",type=str)
parser.add_argument("--basedirvalidation",action="store",type=str)
parser.add_argument("--basedirtest",action="store",type=str)
parser.add_argument("--tbpath", action="store",type=str)
parser.add_argument("--gpu", action="store", type=int)
parser.add_argument("--shuffle", action="store", type=int)
parser.add_argument("--identifier",action="store", type=str)

args = parser.parse_args()

trainDataset = Mapdataset(args.basedirtrain, args.gpu, 58000)
trainLoader = torch.utils.data.DataLoader(trainDataset,args.batchsize,args.shuffle)

validationDataset = Mapdataset(args.basedirvalidation, args.gpu, 1000)
validationLoader = torch.utils.data.DataLoader(validationDataset,args.batchsize,args.shuffle)

testDataset = Mapdataset(args.basedirtest, args.gpu,1)
testLoader = torch.utils.data.DataLoader(testDataset,args.batchsize,args.shuffle)

tensorboard_path = os.path.join(args.tbpath,args.identifier)

writer = SummaryWriter(tensorboard_path)
print("Tensorboard enviorment created at:", tensorboard_path)


if args.gpu:
    print("You running your model at gpu")
    model = UNet(1,1).cuda()
else:
    print("You running your model at cpu")
    model = UNet(1,1)


optimizer = torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=0.999)

ssim = torch.nn.MSELoss() # SSIM(device="cuda:0" if args.gpu else "cpu:0")

val_x , val_y = validationLoader.dataset[0]
writer.add_image("input_image0",val_x,0)
writer.add_image("label_image0",val_y,0)

val_x , val_y = validationLoader.dataset[20]
writer.add_image("input_image20",val_x,0)
writer.add_image("label_image20",val_y,0)

val_x , val_y = validationLoader.dataset[100]
writer.add_image("input_image100",val_x,0)
writer.add_image("label_image100",val_y,0)

def validationStep(model,loader,epoch):
    validationLoss = []
    for validation_x, validation_y in loader:
        val_pred = model(validation_x)
        val_loss = ssim(val_pred,validation_y)
        validationLoss.append(val_loss.item())
    writer.add_scalar("validation_loss",np.mean(validationLoss),epoch)
    val_x , _ = loader.dataset[0]
    val_pred = model(val_x.unsqueeze(0))
    writer.add_image("validation_image0",val_pred[0],epoch)
    val_x , _ = loader.dataset[20]
    val_pred = model(val_x.unsqueeze(0))
    writer.add_image("validation_image20",val_pred[0],epoch)
    val_x , _ = loader.dataset[100]
    val_pred = model(val_x.unsqueeze(0))
    writer.add_image("validation_image100",val_pred[0],epoch)
    return np.mean(validationLoss)

def save_checkpoint(model, optimizer, path, epoch):
   state = {
       'model': model.state_dict(),
       'optimizer': optimizer.state_dict(),
   }
   torch.save(state, path + '_' + str(epoch))

# training loop 
reg_lambda = 0.05
for epoch in range(args.epochs):
    epoch_loss = []
    for train_x, train_y in tqdm(trainLoader, desc=f"epoch = {epoch}"):
            optimizer.zero_grad()
            prediction = model(train_x)            
            loss = ssim(prediction,train_y)             
            loss.backward()
            optimizer.step()
#    tools.saveimage(prediction, train_y,train_x,epoch)
#    print("Loss at Epoch",epoch,":",loss.item())
    epoch_loss.append(loss.item())
        
    if epoch % 1 == 0: 
        eLoss = np.mean(epoch_loss)
        writer.add_scalar('training_loss',loss,epoch)
        val_loss = validationStep(model,validationLoader,epoch)
        model_path = "checkpoints/"+args.identifier+"/ckpt"
        tools.create_path_if_not_exists(model_path)
        save_checkpoint(model,optimizer,model_path,epoch)
        print("Training Loss", eLoss, "Validation Loss", val_loss)




