
from PIL import Image
import numpy as np
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
    def __init__(self, h, w):
        self.h = h
        self.w = w

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
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        min_size = max(img.shape[:2])

        img = A.Resize(self.h,self.w,p=1)(image=img)['image']
        img = Image.fromarray(img)
        return img



######## 2.dataloader



class TensorDatasetTrainClassify(Dataset):

    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])

        if self.transform is not None:
            img = self.transform(img)

        path_dir = '/'.join(self.train_jpg[index].split('/')[:-1])
        if 'true' in path_dir and 'fake' in path_dir: 
            raise Exception("wrong img path "+path_dir)

        y = 0
        if  'true' in self.train_jpg[index]:
            y = 1

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


def getDataLoader(mode, input_data,model_name, img_size, batch_size, kwargs):
    num_workers = 4

    if model_name == 'mobilenetv3':
        my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        raise Exception("Not found normalize type!")


    if mode=="trainClassify":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet

        train_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[0],transforms.Compose([
                                
                                TrainDataAug(img_size, img_size),
                                #ImageNetPolicy(),  #autoaug
                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                transforms.ToTensor(),
                                my_normalize,
                                ])),
                        batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[1],transforms.Compose([
                                TestDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize
                                ])),
                        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        return train_loader, val_loader



    elif mode=="trainClassifyOnehot":
        my_dataloader = TensorDatasetTrainClassify
        
        train_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[0],transforms.Compose([
                                TrainDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize,
                                ])),
                        batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[1],transforms.Compose([
                                TestDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize
                                ])),
                        batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
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
                num_workers=kwargs['num_workers'], pin_memory=kwargs['pin_memory']
            )

        return test_loader


    if mode=="trainval":
        my_dataloader = TensorDatasetTrainClassify
        
        #auto aug

        #from .autoaugment import ImageNetPolicy
        # from libs.FastAutoAugment.data import  Augmentation
        # from libs.FastAutoAugment.archive import fa_resnet50_rimagenet

        train_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[0],transforms.Compose([
                                
                                TrainDataAug(img_size, img_size),
                                #ImageNetPolicy(),  #autoaug
                                #Augmentation(fa_resnet50_rimagenet()), #fastaa
                                transforms.ToTensor(),
                                my_normalize,
                                ])),
                        batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[1],transforms.Compose([
                                TestDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize
                                ])),
                        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        return train_loader, val_loader





