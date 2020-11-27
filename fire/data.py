
import os
import random
import numpy as np
from sklearn.model_selection import KFold



from fire.datatools import getDataLoader, getFileNames




class FireData():
    def __init__(self, cfg):
        
        self.cfg = cfg


        self.kwargs = {'num_workers':self.cfg['num_workers'], 'pin_memory': True}

    def getTrainValDataloader(self):

        if self.cfg['val_path'] != '':
            print("[INFO] val_path is not none, use kflod to split train-val data ...")

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
            for fid in range(self.cfg['k_flod']):
                train_index, val_index = next(folds.split(data_names))
                if fid == self.cfg['start_fold']:
                    break

            train_data = data_names[train_index]
            val_data = data_names[val_index]


        input_data = [train_data, val_data]
        train_loader, val_loader = getDataLoader("trainClassify", 
                                                input_data,
                                                self.cfg['model_name'], 
                                                self.cfg['img_size'], 
                                                self.cfg['batch_size'], 
                                                self.kwargs)
        return train_loader, val_loader

    def getTestDataloader(self):
        pass


    def showTrainData(self):
        #show train data finally to exam
        pass

    