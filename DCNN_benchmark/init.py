# import libs
import os
import time 
from time import strftime,gmtime
import json
import time 
import os
import numpy as np
import imageio
from numpy import random
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import urllib.request
# to store results
import pandas as pd

# figure's variables
fig_width = 20
phi = (np.sqrt(5)+1)/2
phi = phi**2
colors = ['b', 'r', 'k','g']

# host & date's variables 
HOST = os.uname()[1]
#datetag = strftime("%Y-%m-%d", gmtime()) 
datetag = '2020-08-27'

#dataset configuration

image_size = 256 # default image resolution
image_sizes = 2**np.arange(6, 10) # resolutions explored

N_images_per_class = 100
#i_labels = random.randint(1000, size=(N_labels)) # Random choice
i_labels = [409, 530, 892, 487, 920, 704, 879, 963, 646, 620 ] # Pre-selected classes
N_labels = len(i_labels)

id_dl = ''
root = 'data'
folder = 'imagenet_classes_100'
path = os.path.join(root, folder) # data path

with open('ImageNet-datasets-downloader/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
labels[0].split(', ')
labels = [label.split(', ')[1].lower().replace('_', ' ') for label in labels]

class_loader = 'ImageNet-datasets-downloader/imagenet_class_info.json'
with open(class_loader, 'r') as fp: # get all the classes on the data_downloader
    name = json.load(fp)

# a reverse look-up-table giving the index of a given label (within the whole set of imagenet labels)
reverse_labels = {}
for i_label, label in enumerate(labels):
    reverse_labels[label] = i_label
# a reverse look-up-table giving the index of a given i_label (within the sub-set of classes)
reverse_i_labels = {}
for i_label, label in enumerate(i_labels):
    reverse_i_labels[label] = i_label

print('-'*24)
# choosing the selected classes for recognition
for i_label in i_labels: 
    print('label', i_label, '=', labels[i_label])
    for key in name:
        if name[key]['class_name'] == labels[i_label]:
            id_dl += key + ' '
print('label IDs = ', id_dl)