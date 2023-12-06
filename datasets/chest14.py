import os

from PIL import Image

import torch
from torch.utils.data import Dataset
import pandas as pd


class Chest14Dataset(Dataset):
    """
    Chest14 Dataset 
    """

    def __init__(self,
                label_csv,
                img_dir,
                train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Effusion'],
                transform=None) -> None:
        super().__init__()
        self.img_csv = pd.read_csv(label_csv)
        self.img_dir = img_dir

        self.img_name = self.img_csv.iloc[:,0]
        self.img_labels = self.img_csv[train_cols]
 
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_name.iloc[idx])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.img_labels.iloc[idx,:], dtype= torch.float32)
        if self.transform:
            image = self.transform(image)

        return image, label
    
if __name__ == '__main__':
    root = '/home/data_root/chest14/'
    testSet = Chest14Dataset(label_csv = '/home/Chest14/Label_Train_Chest14_Full.csv',
                             img_dir = root)
    
    print(testSet[0][1])