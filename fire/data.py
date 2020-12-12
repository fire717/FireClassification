
import os
import random
import numpy as np
from sklearn.model_selection import KFold

import cv2
from torchvision import transforms

from fire.datatools import getDataLoader, getFileNames, TrainDataAug




class FireData():
    def __init__(self, cfg):
        
        self.cfg = cfg


    def getTrainValDataloader(self):

        if self.cfg['val_path'] != '':
            print("[INFO] val_path is not none, not use kflod to split train-val data ...")
            train_data = getFileNames(self.cfg['train_path'])
            train_data.sort(key = lambda x:os.path.basename(x))
            train_data = np.array(train_data)
            random.shuffle(train_data)

            val_data = getFileNames(self.cfg['val_path'])
            if self.cfg['try_to_train_items'] > 0:
                train_data = train_data[:self.cfg['try_to_train_items']]
                val_data = val_data[:self.cfg['try_to_train_items']]

        else:
            print("[INFO] val_path is none, use kflod to split data: k=%d start_fold=%d" % (self.cfg['k_flod'],self.cfg['start_fold']))
            data_names = getFileNames(self.cfg['train_path'])
            print("[INFO] Total images: ", len(data_names))

            data_names.sort(key = lambda x:os.path.basename(x))
            data_names = np.array(data_names)
            random.shuffle(data_names)

            if self.cfg['try_to_train_items'] > 0:
                data_names = data_names[:self.cfg['try_to_train_items']]

            folds = KFold(n_splits=self.cfg['k_flod'], shuffle=False)
            data_iter = folds.split(data_names)
            for fid in range(self.cfg['k_flod']):
                train_index, val_index = next(data_iter)
                if fid == self.cfg['start_fold']:
                    break

            train_data = data_names[train_index]
            val_data = data_names[val_index]


        input_data = [train_data, val_data]
        train_loader, val_loader = getDataLoader("trainval", 
                                                input_data,
                                                self.cfg)
        return train_loader, val_loader


    def getTrainDataloader(self):
        data_names = getFileNames(self.cfg['train_path'])
        print("[INFO] Total images: ", len(data_names))

        input_data = [data_names]
        train_loader = getDataLoader("train", 
                                        input_data,
                                        self.cfg)
        return train_loader

    def getValDataloader(self):
        data_names = getFileNames(self.cfg['val_path'])
        print("[INFO] Total images: ", len(data_names))

        input_data = [data_names]
        train_loader = getDataLoader("val", 
                                        input_data,
                                        self.cfg)
        return train_loader

    def getTestDataloader(self):
        data_names = getFileNames(self.cfg['test_path'])
        input_data = [data_names]
        test_loader = getDataLoader("test", 
                                    input_data,
                                    self.cfg)
        return test_loader


    def showTrainData(self, show_num = 200):
        #show train data finally to exam

        show_dir = "show_img"
        show_path = os.path.join(self.cfg['save_dir'], show_dir)
        if not os.path.exists(show_path):
            os.makedirs(show_path)


        img_path_list = getFileNames(self.cfg['train_path'])[:show_num]
        transform = transforms.Compose([TrainDataAug(self.cfg['img_size'])])


        for i,img_path in enumerate(img_path_list):
            #print(i)
            img = cv2.imread(img_path)
            img = transform(img)
            img.save(os.path.join(show_path,os.path.basename(img_path)), quality=100)

    