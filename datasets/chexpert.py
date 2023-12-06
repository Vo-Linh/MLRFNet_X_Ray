import numpy as np
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
from PIL import Image
import pandas as pd

class CheXpertDataset(Dataset):
    '''
    Reference: 
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''
    def __init__(self, 
                 csv_path, 
                 image_root_path='',
                 ignore_index=0, 
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 verbose=False,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion'],
                 transform = None):
        
        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        self.transform = transform
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']  
            
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print ('Upsampling %s...'%col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # impute missing values 
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)  
                self.df[col].fillna(0, inplace=True) 
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)
        
        self._num_images = len(self.df)
        
        # 0 --> -1
        if flip_label and ignore_index != -1: 
            # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)
        
        assert ignore_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if ignore_index == -1: # 5 classes
            print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:       # 1 class
            self.select_cols = [train_cols[ignore_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()
        
        self.ignore_index = ignore_index
        
        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        if ignore_index != -1:
            self._labels_list = self.df[train_cols].values[:, ignore_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()
    
        if verbose:
            if ignore_index != -1:
                print ('-'*30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], ignore_index, self.imratio ))
                else:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], ignore_index, self.imratio ))
                print ('-'*30)
            else:
                print ('-'*30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                    print ()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print ('-'*30)
            
    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)
       
    @property  
    def data_size(self):
        return self._num_images 
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = Image.open(self._images_list[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        
        if self.ignore_index != -1: # multi-class mode
            label = torch.tensor(self._labels_list[idx]).reshape(-1).astype(np.float32)
        else:
            label = torch.tensor(self._labels_list[idx], dtype= torch.float32)
        return image, label


if __name__ == '__main__':
    root = '/home/data_root/chesxpert/'
    traindSet = CheXpertDataset(csv_path=root+'train.csv', image_root_path=root, use_upsampling=True, use_frontal=True, ignore_index=0)
    testSet =  CheXpertDataset(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, ignore_index=-1)
    print(testSet[4][1])

 