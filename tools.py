import torchvision
import os
def saveimage(prediction,target,input, epoch):
    if not os.path.exists("runs/current/"):
        os.makedirs("runs/current/")
 
    torchvision.utils.save_image(prediction,f'runs/current/{epoch}-prediction.png',normalize=True)
    torchvision.utils.save_image(target,f'runs/current/{epoch}-target.png',normalize=True)
    torchvision.utils.save_image(input,f'runs/current/{epoch}-input.png',normalize=True)