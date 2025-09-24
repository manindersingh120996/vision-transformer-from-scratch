import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
from PIL import Image
from torchvision.transforms import v2
import os

import random
import shutil
from pathlib import Path

import numpy as np
class TomAndJerryDataset(Dataset):
    def __init__(self,dataset_path,transform=None, target_transform=None):
        super().__init__()
        self.transformation = transform

        self.target_transform = target_transform
        self.classes = os.listdir(dataset_path)
        self.class_to_index = {self.classes[ix]:ix for ix in range(len(self.classes))}
        self.index_to_class = {ix:self.classes[ix] for ix in range(len(self.classes))}
        self.images = glob.glob(dataset_path+'/*/*.jpg')
        self.dataset_path = dataset_path
        self.to_tensor = v2.ToTensor()
        # print(self.images[0])


    def __len__(self,):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image).convert('RGB')
        
        # label = str(Path(self.images[index]).parent).split('/')[-1]
        label = Path(self.images[index]).parent.name
        # print(label)
        # print(np.array(image).shape)
        # plt.imshow(image)

        if self.transformation is not None:
            image = self.transformation(image)
        else:
            image = self.to_tensor(image)

        
        label = self.class_to_index[label]
        if self.target_transform:
            label = self.target_transform(label)
        # plt.imshow(image)
        
        return image,label
    

# transform = v2.Compose([
#     v2.Resize((224,224)),
#     v2.RandomVerticalFlip(p=0.34),
#     v2.ToTensor()

# ])

train_transform = v2.Compose([

    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)), 
    v2.TrivialAugmentWide(),   
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

val_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])