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

    model_path = 'output/mobilenetv3_e13_0.99905.pth'
    runner.modelLoad(model_path)

    move_dir = "../data/dataset/tmp"
    target_label = 1
    runner.cleanData(train_loader, target_label, move_dir)



if __name__ == '__main__':
    main(cfg)