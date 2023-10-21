import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import pandas as pd


class Chest14Dataset(Dataset):
    """
    Chest14 Dataset 
    """

    def __init__(self, label_csv, img_dir, transform=None) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(label_csv)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.img_labels.iloc[idx,1:], dtype= torch.float32)
        image = self.transform(image)
            
        return image, label