import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg




def main(cfg):


    initFire(cfg)


    model = FireModel(cfg)
    #print(model)



    

    data = FireData(cfg)
    # data.showTrainData()
    # b
    
    train_loader, val_loader = data.getTrainValDataloader()


    runner = FireRunner(cfg, model)
    runner.train(train_loader, val_loader)


    ## test
    #test_loader = data.getTestDataloader()
    #runner.test(test_loader)


if __name__ == '__main__':
    main(cfg)