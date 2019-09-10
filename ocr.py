try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import torch
from models.unet import UNet
from dataset import Mapdataset
import matplotlib.pyplot as plt

# make shure you installed german language data for tesseract
# sudo apt-get install tesseract-ocr-deu 
#print(pytesseract.image_to_string(Image.open('docs/sample.jpeg'), lang='deu'))

model = UNet(1,1)
model.load_state_dict(torch.load("checkpoints/ckpt_29",map_location="cpu")["model"])

image = Mapdataset.get_image("docs/Real_map_data/Waldfriede.jpeg")
model.eval()

output = model.forward(image.unsqueeze(0))

plt.imshow(output[0,0].detach().numpy(),cmap="gray")
plt.axis('off')
plt.savefig("output.png")

print(pytesseract.image_to_string(Image.open('output.png')))