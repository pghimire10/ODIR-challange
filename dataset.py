from torch.utils.data import Dataset
from os.path import join


from utils.utility import read_as_csv
from transforms import  labels_to_idx
from PIL import Image





class ImageDataset(Dataset):
    def __init__(self, csv_path, transform):
        super().__init__()

        images, labels = read_as_csv(csv_path)

        self.images = images
        self.labels = labels
        self.transform = transform

    
    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, index):
        image_name =  self.images[index]
        label_name = self.labels[index]
        image_path = join("ODIR-5K_Training_Images\ODIR-5K_Training_Dataset", image_name)
        image = read_image(image_path)

        if self.transform:
            image= self.transform(image)

        
        label = labels_to_idx(label_name)

        return image, label


def read_image(img_path):
    img = Image.open(img_path).convert("RGB")
    return img.copy()

