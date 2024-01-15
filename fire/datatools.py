 
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import cv2
import albumentations as A
import json
import platform

from fire.utils import firelog
from fire.dataaug_user import TrainDataAug, TestDataAug


##### Common
def getFileNames(file_dir, tail_list=['.png','.jpg','.JPG','.PNG']): 
        L=[] 
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] in tail_list:
                    L.append(os.path.join(root, file))
        return L




######## dataloader

class TensorDatasetTrainClassify(Dataset):
    def __init__(self, data, cfg, transform=None):
        self.data = data
        self.cfg = cfg
        self.transform = transform


    def __getitem__(self, index):

        img = cv2.imread(self.data[index][0])
        img = cv2.resize(img, self.cfg['img_size'])

        if self.transform is not None:
            img = self.transform(img)
        
        y = self.data[index][1]

        # y_onehot = [0,0]
        # y_onehot[y] = 1

        return img, y, self.data[index]
        
    def __len__(self):
        return len(self.data)


class TensorDatasetTestClassify(Dataset):

    def __init__(self, data, cfg, transform=None):
        self.data = data
        self.cfg = cfg
        self.transform = transform

    def __getitem__(self, index):

        img = cv2.imread(self.data[index])
        img = cv2.resize(img, self.cfg['img_size'])
        #img = imgPaddingWrap(img)
        #b
        if self.transform is not None:
            img = self.transform(img)

        # path_dir = '/'.join(self.data[index].split('/')[:-1])
        # y = 0
        # if  'true' in path_dir:
        #     y = 1

        return img, self.data[index]

    def __len__(self):
        return len(self.data)


###### 3. get data loader 


def getNormorlize(model_name):
    if model_name in ['mobilenetv2','mobilenetv3']:
        my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif model_name == 'xception':
        my_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif "adv-eff" in model_name:
        my_normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    elif "resnex" in model_name or 'eff' in model_name or 'RegNet' in model_name:
        my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #my_normalize = transforms.Normalize([0.4783, 0.4559, 0.4570], [0.2566, 0.2544, 0.2522])
    elif "EN-B" in model_name:
        my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        firelog("i","Not set normalize type, Use defalut imagenet normalization.")
        my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return my_normalize


def getDataLoader(mode, input_data, cfg):

    my_normalize = getNormorlize(cfg['model_name'])



    data_aug_train = TrainDataAug(cfg['img_size'])
    data_aug_test = TestDataAug(cfg['img_size'])


    if mode=="test":
        my_dataloader = TensorDatasetTestClassify

        test_loader = torch.utils.data.DataLoader(
                my_dataloader(input_data[0],
                                cfg,
                                transforms.Compose([
                                    data_aug_test,
                                    transforms.ToTensor(),
                                    my_normalize
                                ])
                ), batch_size=cfg['test_batch_size'], shuffle=False, 
                num_workers=cfg['num_workers'], pin_memory=True
            )

        return test_loader


    elif mode=="trainval":
        my_dataloader = TensorDatasetTrainClassify
        
        train_loader = torch.utils.data.DataLoader(
                                        my_dataloader(input_data[0],
                                            cfg,
                                            transforms.Compose([
                                                data_aug_train,
                                                #ImageNetPolicy(),  #autoaug
                                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                                transforms.ToTensor(),
                                                my_normalize,
                                        ])),
                                batch_size=cfg['batch_size'], 
                                shuffle=True, 
                                num_workers=cfg['num_workers'],
                                pin_memory=True)


        val_loader = torch.utils.data.DataLoader(
                                        my_dataloader(input_data[1],
                                            cfg,
                                            transforms.Compose([
                                                data_aug_test,
                                                transforms.ToTensor(),
                                                my_normalize
                                        ])),
                                batch_size=cfg['batch_size'], 
                                shuffle=False, 
                                num_workers=cfg['num_workers'], 
                                pin_memory=True)
        return train_loader, val_loader
