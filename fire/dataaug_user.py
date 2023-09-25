 
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




###### 1.Data aug
class TrainDataAug:
    def __init__(self, img_size):
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # raw_h, raw_w = img.shape[:2]
        # min_size = max(img.shape[:2])


  
        # img = A.OneOf([A.ShiftScaleRotate(
        #                         shift_limit=0.1,
        #                         scale_limit=0.1,
        #                         rotate_limit=30,
        #                         interpolation=cv2.INTER_LINEAR,
        #                         border_mode=cv2.BORDER_CONSTANT,
        #                          value=0, mask_value=0,
        #                         p=0.5),
        #                 A.GridDistortion(num_steps=5, distort_limit=0.2,
        #                     interpolation=1, border_mode=4, p=0.4),
        #                 A.RandomGridShuffle(grid=(3, 3),  p=0.3)],
        #                 p=0.5)(image=img)['image']

        # img = A.HorizontalFlip(p=0.5)(image=img)['image'] 
        # img = A.VerticalFlip(p=0.4)(image=img)['image'] 
        
        # # img = A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.05, 
        # #                                        contrast_limit=0.05, p=0.5), 
        # #                 A.HueSaturationValue(hue_shift_limit=10, 
        # #                     sat_shift_limit=10, val_shift_limit=10,  p=0.5)], 
        # #                 p=0.4)(image=img)['image']


        # # img = A.GaussNoise(var_limit=(5.0, 10.0), mean=0, p=0.2)(image=img)['image']


        # img = A.RGBShift(r_shift_limit=5,
        #                     g_shift_limit=5,
        #                     b_shift_limit=5,
        #                     p=0.5)(image=img)['image']

        
        # img = A.Resize(self.h,self.w,cv2.INTER_LANCZOS4,p=1)(image=img)['image']
        # img = A.OneOf([A.GaussianBlur(blur_limit=3, p=0.1),
        #                 A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        #                 A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.4)], 
        #                 p=0.4)(image=img)['image']

        # img = A.CoarseDropout(max_holes=3, max_height=20, max_width=20, 
        #                     p=0.8)(image=img)['image']



        
        #img = Image.fromarray(img)
        return img


class TestDataAug:
    def __init__(self, img_size):
        self.h = img_size[0]
        self.w = img_size[1]

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


        # img = A.Resize(self.h,self.w,cv2.INTER_LANCZOS4,p=1)(image=img)['image']
        #img = Image.fromarray(img)
        return img



