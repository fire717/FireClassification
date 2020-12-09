import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd



def main(cfg):

    cfg['test_batch_size'] = 1
    initFire(cfg)


    model = FireModel(cfg)
    # print(model)
    # b

    data = FireData(cfg)
    # data.showTrainData()
    # b
    
    test_loader = data.getTestDataloader()


    runner = FireRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'], data_parallel = False)

    show_count = 1
    runner.heatmap(test_loader, cfg["save_dir"], show_count)



    



if __name__ == '__main__':
    main(cfg)