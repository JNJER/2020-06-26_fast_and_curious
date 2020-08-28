
from DCNN_benchmark.init import *

import torch
import torchvision
import torchvision.transforms as transforms

# transform function for input's image processing
transform = transforms.Compose([
    transforms.Resize(int(image_size)),      # Resize the image to image_size x image_size pixels size.
    transforms.CenterCrop(int(image_size-20)),  # Crop the image to (image_size-20) x (image_size-20) pixels around the center.
    transforms.ToTensor(),       # Convert the image to PyTorch Tensor data type.
    transforms.Normalize(        # Normalize the image by adjusting its average and
                                 #     its standard deviation at the specified values.
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
    )])


image_dataset = ImageFolder(path, transform=transform) # save the dataset

# imports networks with weights
models = {} # get model's names

models['alex'] = torchvision.models.alexnet(pretrained=True)
models['vgg'] = torchvision.models.vgg16(pretrained=True)
models['mob'] = torchvision.models.mobilenet_v2(pretrained=True)
models['res'] = torchvision.models.resnext101_32x8d(pretrained=True)


# Select a device (CPU or CUDA)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for name in models.keys():
    models[name].to(device)
