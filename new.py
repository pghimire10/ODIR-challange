from torch.utils.data import Dataset
from os.path import join
import pandas as pd
import csv
from utils.utility import read_csv
from transforms import transforms
from pil import Image

label_to_index = {'N':0,
   'D':1,
   'G':2,
    'C':3,
    'A':4,
    'H':5,
    'M':6,
    '0':7}

class ImageDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        images, labels = read_csv(csv_path)
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image_name = self.images[index]
        label_name = self.labels[index]
        image_path= join("Data", "ODIR-5K_Training_Dataset", image_name)
        
        image = Image.open(image_path).convert("RGB")
        label = label_to_index[label_name]
        
        if self.transforms:
            image = self.transforms(image)

        return image, label