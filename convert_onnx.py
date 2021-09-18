import os,argparse
import random
import torch        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd



def main(cfg):


    initFire(cfg)


    model = FireModel(cfg)
    
    

    data = FireData(cfg)
    # data.showTrainData()
    # b
    
    test_loader = data.getTestDataloader()


    runner = FireRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'])


    runner.model.eval()
    runner.model.to("cuda")

    #data type nchw
    dummy_input1 = torch.randn(1, 3, 224, 224).to("cuda")
    input_names = [ "input1"] #自己命名
    output_names = [ "output1"]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, "output/model.onnx", 
        verbose=True, input_names=input_names, output_names=output_names,
        do_constant_folding=True)




if __name__ == '__main__':
    main(cfg)