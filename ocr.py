try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import torch
from models.unet import UNet

# make shure you installed german language data for tesseract
# sudo apt-get install tesseract-ocr-deu 
#print(pytesseract.image_to_string(Image.open('docs/sample.jpeg'), lang='deu'))

model = UNet(1,1)
model.load_state_dict(torch.load("")["model"])
model.eval()



print(pytesseract.image_to_string(Image.open('docs/Fdkah.png'),))