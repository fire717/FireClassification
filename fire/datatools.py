 
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
    _print_times = 0
    def __init__(self, train_jpg, label_type, label_path, log_classname=True, transform=None):
        self.train_jpg = train_jpg
        self.label_type = label_type
        self.label_path = label_path
        self.transform = transform
        self.log_classname = log_classname

        self.label_dict = {}
        self.getLabels()
        self.cate_dirs = []

    def getLabels(self):

        if self.label_type == "DIR":
            self.cate_dirs = os.listdir(self.label_path)
            self.cate_dirs.sort()
            if TensorDatasetTrainClassify._print_times==0:
                if self.log_classname:
                    print("[INFO] Default classes names: ", self.cate_dirs)
                    TensorDatasetTrainClassify._print_times=1
            
            for i, img_path in enumerate(self.train_jpg):
                img_dirs = img_path.replace(self.label_path,'')
                img_dirs = img_dirs.split('/')[:2]
                img_dir = img_dirs[0] if img_dirs[0] else img_dirs[1]

                y = self.cate_dirs.index(img_dir)
                self.label_dict[img_path] = y

        # User code here
        elif self.label_type == "CSV":
            df = pd.read_csv(self.label_path)
            dir_path = os.path.dirname(self.train_jpg[0])
            for index, row in df.iterrows():
                #print(row["image_id"], type(row["label"]))
                img_path = os.path.join(dir_path, row["image_id"])
                img_path = img_path.replace("\\","/")
                # print(img_path)
                # b
                self.label_dict[img_path] = row["label"]
                #b
            if TensorDatasetTrainClassify._print_times==0:
                print("[INFO] Labels count: ")
                print(df['label'].value_counts().sort_index())
                TensorDatasetTrainClassify._print_times=1

        else:
            raise Exception("[ERROR] In datatools.py getLabel() reimplement needed. ")
        


    def __getitem__(self, index):

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])

        if self.transform is not None:
            img = self.transform(img)

        y = self.label_dict[self.train_jpg[index]]

        # y_onehot = [0,0]
        # y_onehot[y] = 1

        return img, y, self.train_jpg[index]
        
    def __len__(self):
        return len(self.train_jpg)


class TensorDatasetTestClassify(Dataset):

    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # img = cv2.imread(self.train_jpg[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img,(180, 180))

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])
        #img = imgPaddingWrap(img)
        #b
        if self.transform is not None:
            img = self.transform(img)

        # path_dir = '/'.join(self.train_jpg[index].split('/')[:-1])
        # y = 0
        # if  'true' in path_dir:
        #     y = 1

        return img, self.train_jpg[index]

    def __len__(self):
        return len(self.train_jpg)


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
        raise Exception("[ERROR] Not found normalize type!")
    return my_normalize


def getDataLoader(mode, input_data, cfg):

    my_normalize = getNormorlize(cfg['model_name'])



    if mode=="trainval_onehot":
        my_dataloader = TensorDatasetTrainClassify
        
        train_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[0],transforms.Compose([
                                TrainDataAug(cfg['img_size']),
                                transforms.ToTensor(),
                                my_normalize,
                                ])),
                        batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        val_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[1],transforms.Compose([
                                TestDataAug(cfg['img_size']),
                                transforms.ToTensor(),
                                my_normalize
                                ])),
                        batch_size=1, shuffle=False, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])
        return train_loader, val_loader


    elif mode=="test":
        my_dataloader = TensorDatasetTestClassify

        test_loader = torch.utils.data.DataLoader(
                my_dataloader(input_data[0],
                        transforms.Compose([
                                    TestDataAug(cfg['img_size']),
                                    transforms.ToTensor(),
                                    my_normalize
                                ])
                ), batch_size=cfg['test_batch_size'], shuffle=False, 
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )

        return test_loader


    elif mode=="trainval":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet
        if cfg['label_type'] == 'DIR':
            cfg['label_path'] = cfg['train_path']

        train_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[0],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            True,
                                            transforms.Compose([
                                                TrainDataAug(cfg['img_size']),
                                                #ImageNetPolicy(),  #autoaug
                                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                                transforms.ToTensor(),
                                                my_normalize,
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        if cfg['val_path']:
            cfg['label_path'] = cfg['val_path']

        val_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[1],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            True,
                                            transforms.Compose([
                                                TestDataAug(cfg['img_size']),
                                                transforms.ToTensor(),
                                                my_normalize
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])
        return train_loader, val_loader


    elif mode=="train":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet
        if cfg['label_type'] == 'DIR':
            cfg['label_path'] = cfg['train_path']

        train_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[0],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            False,
                                            transforms.Compose([
                                                TestDataAug(cfg['img_size']),
                                                #ImageNetPolicy(),  #autoaug
                                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                                transforms.ToTensor(),
                                                my_normalize,
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        return train_loader

    elif mode=="val":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet
        if cfg['label_type'] == 'DIR':
            cfg['label_path'] = cfg['val_path']

        train_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[0],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            False,
                                            transforms.Compose([
                                                TestDataAug(cfg['img_size']),
                                                #ImageNetPolicy(),  #autoaug
                                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                                transforms.ToTensor(),
                                                my_normalize,
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        return train_loader

