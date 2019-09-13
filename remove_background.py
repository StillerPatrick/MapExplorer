try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import torch
from models.unet import UNet
from dataset import Mapdataset
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy
#from torchvision import transforms

# make shure you installed german language data for tesseract
# sudo apt-get install tesseract-ocr-deu 
#print(pytesseract.image_to_string(Image.open('docs/sample.jpeg'), lang='deu'))

def array_to_img(data,path):
    new_im = Image.fromarray(data*255).convert('LA')
    #if new_im.mode != 'RGB':
    #    new_im = new_im.convert('RGB')
    new_im.save(path)

def clean_image(path,output_dir):
    image = Mapdataset.get_image(path)
    output =model.forward(image.unsqueeze(0))
    data =output[0,0].detach().numpy()
    out_path = output_dir + os.path.basename(path) + ".png"
    array_to_img(data,out_path)
    #plt.figure(figsize=(400, 75), dpi=1)
    #plt.imshow(data,cmap="gray", aspect='equal')
    #plt.axis('off')
    #plt.savefig(out_path)

model = UNet(1,1)
model.load_state_dict(torch.load("checkpoints/data_56k_lr4_bin_dice2/ckpt_1",map_location="cpu")["model"])
model.eval()
folder = "docs/Real_map_data/"
print("converting")
for image in os.listdir(folder):
    print(image)
    clean_image(folder + image,"out/")

#print(pytesseract.image_to_string(Image.open('output.png')))