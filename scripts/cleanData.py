import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg




def main(cfg):


    initFire(cfg)


    model = FireModel(cfg)
    
    

    data = FireData(cfg)
    # data.showTrainData()
    # b
    
    train_loader = data.getTrainDataloader()


    runner = FireRunner(cfg, model)

    runner.modelLoad(cfg['model_path'])

    move_dir = '../data/dataset/d_trainval/v8/tmp'
    #"../data/dataset/a_raw_data/3_our/0_bigimg/add/antispoof20201214/true"
    #"../data/dataset/d_trainval/v8/tmp"
    target_label = 1
    runner.cleanData(train_loader, target_label, move_dir)



if __name__ == '__main__':
    main(cfg)