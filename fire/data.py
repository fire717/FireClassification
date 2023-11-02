
import os
import random
import numpy as np

import cv2
from torchvision import transforms

from fire.utils import firelog
from fire.datatools import getDataLoader, getFileNames
from fire.dataaug_user import TrainDataAug



class FireData():
    def __init__(self, cfg):
        self.cfg = cfg


    def getTrainValDataloader(self):

        class_names = self.cfg['class_names']
        if len(class_names)==0:
            class_names = os.listdir(self.cfg['train_path'])
            class_names.sort()
        firelog("i", class_names)

        train_data = []
        for i,class_name in enumerate(class_names):
            sub_dir = os.path.join(self.cfg['train_path'],class_name)
            img_path_list = getFileNames(sub_dir)
            img_path_list.sort()
            train_data += [[p,i] for p in img_path_list]
        random.shuffle(train_data)

        if self.cfg['val_path'] != '':
            firelog('i',"val_path is not none, not use kflod to split train-val data ...")
            
            val_data = []
            for i,class_name in enumerate(class_names):
                sub_dir = os.path.join(self.cfg['val_path'],class_name)
                img_path_list = getFileNames(sub_dir)
                img_path_list.sort()
                val_data += [[p,i] for p in img_path_list]

        else:
            firelog('i',"val_path is none, use kflod to split data: k=%d val_fold=%d" % (self.cfg['k_flod'],self.cfg['val_fold']))
            all_data = train_data

            fold_count = int(len(all_data)/self.cfg['k_flod'])
            if self.cfg['val_fold']==self.cfg['k_flod']:
                train_data = all_data
                val_data = all_data[:10]
            else:
                val_data = all_data[fold_count*self.cfg['val_fold']:fold_count*(self.cfg['val_fold']+1)]
                train_data = all_data[:fold_count*self.cfg['val_fold']]+all_data[fold_count*(self.cfg['val_fold']+1):]

        if self.cfg['try_to_train_items'] > 0:
            train_data = train_data[:self.cfg['try_to_train_items']]
            val_data = val_data[:self.cfg['try_to_train_items']]

        firelog('i',"Train: %d Val: %d " % (len(train_data),len(val_data)))
        input_data = [train_data, val_data]

        train_loader, val_loader = getDataLoader("trainval", 
                                                input_data,
                                                self.cfg)
        return train_loader, val_loader


    def getEvalDataloader(self):
        data_names = getFileNames(self.cfg['eval_path'])
        firelog('i',"Total images: "+str(len(data_names)))

        input_data = [data_names]
        data_loader = getDataLoader("eval", 
                                        input_data,
                                        self.cfg)
        return data_loader

    def getTestDataloader(self):
        data_names = getFileNames(self.cfg['test_path'])
        input_data = [data_names]
        data_loader = getDataLoader("test", 
                                    input_data,
                                    self.cfg)
        return data_loader


    def showTrainData(self, show_num = 200):
        #show train data finally to exam

        show_dir = "show_img"
        show_path = os.path.join(self.cfg['save_dir'], show_dir)
        firelog('i',"Showing traing data in ",show_path)
        if not os.path.exists(show_path):
            os.makedirs(show_path)


        img_path_list = getFileNames(self.cfg['train_path'])[:show_num]
        transform = transforms.Compose([TrainDataAug(self.cfg['img_size'])])


        for i,img_path in enumerate(img_path_list):
            #print(i)
            img = cv2.imread(img_path)
            img = transform(img)
            img.save(os.path.join(show_path,os.path.basename(img_path)), quality=100)

    