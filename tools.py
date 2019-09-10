import torchvision
import os
def saveimage(prediction,target, epoch):
    if not os.path.exists("runs/current/"):
        os.makedirs("runs/current/")
 
    torchvision.utils.save_image(prediction[0],f'runs/current/{epoch}-prediction.png')
    torchvision.utils.save_image(target[0],f'runs/current/{epoch}-target.png',normalize=True)