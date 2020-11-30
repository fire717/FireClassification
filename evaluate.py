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


    runner.evaluate(train_loader)



if __name__ == '__main__':
    main(cfg)