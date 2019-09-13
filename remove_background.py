import torch
from models.unet import UNet
from models.tiramisu import FCDenseNet103
from dataset import Mapdataset
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy

def array_to_img(data,path):
    new_im = Image.fromarray(data*255).convert('LA')
    new_im.save(path)

def clean_image(path,output_dir):
    image = Mapdataset.get_image(path)
    output =model.forward(image.unsqueeze(0))
    data =output[0,0].detach().numpy()
    out_path = output_dir + os.path.basename(path) + ".png"
    array_to_img(data,out_path)

model = UNet(1,1)
model.load_state_dict(torch.load("checkpoints/data_56k_lr4_bin_dice2/ckpt_1",map_location="cpu")["model"])
model.eval()
folder = "Ausschnitte/"
print("converting")
for image in os.listdir(folder):
    print(image)
    clean_image(path=folder+image, output_dir="out/")
