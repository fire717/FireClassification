 
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




##### Common
def getFileNames(file_dir, tail_list=['.png','.jpg','.JPG','.PNG']): 
        L=[] 
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] in tail_list:
                    L.append(os.path.join(root, file))
        return L


###### 1.Data aug
class TrainDataAug:
    def __init__(self, img_size):
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        raw_h, raw_w = img.shape[:2]
        min_size = max(img.shape[:2])


  
        img = A.ShiftScaleRotate(
                                shift_limit=0.1,
                                scale_limit=0.1,
                                rotate_limit=10,
                                interpolation=cv2.INTER_LINEAR,
                                border_mode=cv2.BORDER_CONSTANT,
                                 value=0, mask_value=0,
                                p=0.5)(image=img)['image']

        img = A.HorizontalFlip(p=0.5)(image=img)['image'] 
        
        img = A.OneOf([A.RandomBrightness(limit=0.1, p=1), 
                    A.RandomContrast(limit=0.1, p=1),
                    A.RandomGamma(gamma_limit=(50, 150),p=1),
                    A.HueSaturationValue(hue_shift_limit=10, 
                        sat_shift_limit=10, val_shift_limit=10,  p=1)], 
                    p=0.6)(image=img)['image']

        
        img = A.Resize(self.h,self.w,p=1)(image=img)['image']
        img = A.OneOf([A.MotionBlur(blur_limit=3, p=0.2), 
                        A.MedianBlur(blur_limit=3, p=0.2), 
                        A.GaussianBlur(blur_limit=3, p=0.1),
                        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5)], 
                        p=0.8)(image=img)['image']

        img = A.CoarseDropout(max_holes=3, max_height=20, max_width=20, 
                            p=1)(image=img)['image']



        
        img = Image.fromarray(img)
        return img


class TestDataAug:
    def __init__(self, img_size):
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


        img = A.Resize(self.h,self.w,p=1)(image=img)['image']
        img = Image.fromarray(img)
        return img



######## 2.dataloader



class TensorDatasetTrainClassify(Dataset):
    _print_times = 0
    def __init__(self, train_jpg, label_type, label_path, transform=None):
        self.train_jpg = train_jpg
        self.label_type = label_type
        self.label_path = label_path
        self.transform = transform

        self.label_dict = {}
        self.getLabels()
        self.cate_dirs = []

    def getLabels(self):

        if self.label_type == "DIR":
            self.cate_dirs = os.listdir(self.label_path)
            self.cate_dirs.sort()
            print("[INFO] Default classes names: ", self.cate_dirs)
            
            for i, img_path in enumerate(self.train_jpg):
                #print(self.label_path)
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
                #print(dir_path,img_path)
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

        return img, y
        
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
    if model_name == 'mobilenetv3':
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
                        batch_size=batch_size, shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

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
                                    TestDataAug(img_size, img_size),
                                    transforms.ToTensor(),
                                    my_normalize
                                ])
                ), batch_size=batch_size, shuffle=False, 
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )

        return test_loader


    elif mode=="trainval":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet
        if cfg['label_path'] == 'DIR':
            cfg['label_path'] = cfg['train_path']

        train_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[0],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            transforms.Compose([
                                                TrainDataAug(cfg['img_size']),
                                                #ImageNetPolicy(),  #autoaug
                                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                                transforms.ToTensor(),
                                                my_normalize,
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

        val_loader = torch.utils.data.DataLoader(
                                my_dataloader(input_data[1],
                                            cfg['label_type'],
                                            cfg['label_path'],
                                            transforms.Compose([
                                                TestDataAug(cfg['img_size']),
                                                transforms.ToTensor(),
                                                my_normalize
                                                ])),
                                batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])
        return train_loader, val_loader





