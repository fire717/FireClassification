import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np


CATES1 = ['LongSleeve', 'ShortSleeve', 'NoSleeve']
CATES2 = ['Solidcolor', 'multicolour', 'lattice']
CATES3 = ['Short', 'Long', 'middle', 'Bald']
CATES4 = ['Skirt', 'Trousers', 'Shorts']
CATES5 = ['Solidcolor', 'multicolour', 'lattice']
CATES6 = ['Sandals', 'Sneaker', 'LeatherShoes', 'else']
CATES7 = ['left', 'right', 'back', 'front']


def flipRes(d):
    for k,v in d.items():
        # print(d[k][2]) 
        d[k][6][0],d[k][6][1] = d[k][6][1],d[k][6][0]
        # print(d[k][2]) 
        # b
    return d


def _colorDecode(color):
    for j in range(len(color)):
        new_c =  int(color[j]*10+0.5)
        color[j]=new_c
    if sum(color)<10:
        #print(color[np.argmax(color)],sum(color),color)
        color[np.argmax(color)] += 10-int(sum(color))
        #print(color[np.argmax(color)])
        
    color /=10
    return color


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def main(cfg):


    initFire(cfg)


    model = FireModel(cfg)
    data = FireData(cfg)
    # data.showTrainData()
    # b
    
    test_loader = data.getTestDataloader()
    test_loader2 = data.getTestDataloaderFlip()
    runner = FireRunner(cfg, model)
    runner.modelLoad('output/convnext_large_22k_1k_384_e5_fold0_0.75868.pth')
    print("load model1, start running.")
    model1_res_dict1 = runner.predictRaw(test_loader)
    model1_res_dict2 = runner.predictRaw(test_loader2)
    model1_res_dict2 = flipRes(model1_res_dict2)

    test_loader = data.getTestDataloader()
    test_loader2 = data.getTestDataloaderFlip()
    runner = FireRunner(cfg, model)
    runner.modelLoad('output/convnext_large_22k_1k_384_e14_fold0_0.75514.pth')
    print("load model2, start running.")
    model2_res_dict1 = runner.predictRaw(test_loader)
    model2_res_dict2 = runner.predictRaw(test_loader2)
    model2_res_dict2 = flipRes(model2_res_dict2)

    test_loader = data.getTestDataloader()
    test_loader2 = data.getTestDataloaderFlip()
    runner = FireRunner(cfg, model)
    runner.modelLoad('output/convnext_large_22k_1k_384_e15_fold0_0.75059.pth')
    print("load model3, start running.")
    model3_res_dict1 = runner.predictRaw(test_loader)
    model3_res_dict2 = runner.predictRaw(test_loader2)
    model3_res_dict2 = flipRes(model3_res_dict2)




    res_dict = {}
    for k,v in model1_res_dict1.items():
        #print(k,v)
        v1 = CATES1[np.argmax(v[0]+model1_res_dict2[k][0]+model2_res_dict1[k][0]+model2_res_dict2[k][0]+model3_res_dict1[k][0]+model3_res_dict2[k][0])]
        v2 = CATES2[np.argmax(v[1]+model1_res_dict2[k][1]+model2_res_dict1[k][1]+model2_res_dict2[k][1]+model3_res_dict1[k][1]+model3_res_dict2[k][1])]
        v3 = CATES3[np.argmax(v[2]+model1_res_dict2[k][2]+model2_res_dict1[k][2]+model2_res_dict2[k][2]+model3_res_dict1[k][2]+model3_res_dict2[k][2])]
        v4 = CATES4[np.argmax(v[3]+model1_res_dict2[k][3]+model2_res_dict1[k][3]+model2_res_dict2[k][3]+model3_res_dict1[k][3]+model3_res_dict2[k][3])]
        v5 = CATES5[np.argmax(v[4]+model1_res_dict2[k][4]+model2_res_dict1[k][4]+model2_res_dict2[k][4]+model3_res_dict1[k][4]+model3_res_dict2[k][4])]
        v6 = CATES6[np.argmax(v[5]+model1_res_dict2[k][5]+model2_res_dict1[k][5]+model2_res_dict2[k][5]+model3_res_dict1[k][5]+model3_res_dict2[k][5])]
        v7 = CATES7[np.argmax(v[6]+model1_res_dict2[k][6]+model2_res_dict1[k][6]+model2_res_dict2[k][6]+model3_res_dict1[k][6]+model3_res_dict2[k][6])]
        v8 = softmax(v[7]+model1_res_dict2[k][7]+model2_res_dict1[k][7]+model2_res_dict2[k][7]+model3_res_dict1[k][7]+model3_res_dict2[k][7])
        v9 = softmax(v[8]+model1_res_dict2[k][8]+model2_res_dict1[k][8]+model2_res_dict2[k][8]+model3_res_dict1[k][8]+model3_res_dict2[k][8])

        v8 = _colorDecode(v8)
        v9 = _colorDecode(v9)   
        res_dict[k] = [v1,v2,v3,
                        v4,v5,v6,
                        v7,*v8,*v9]
    
    res_list = sorted(res_dict.items(), key = lambda kv: int(kv[0].split("_")[-1].split('.')[0]))
    #print(len(res_list), res_list[0])
    # to csv
    with open('result.csv', 'w', encoding='utf-8') as f:
        f.write('name,upperLength,clothesStyles,hairStyles,lowerLength,lowerStyles,shoesStyles,towards,upperBlack,upperBrown,upperBlue,upperGreen,upperGray,upperOrange,upperPink,upperPurple,upperRed,upperWhite,upperYellow,lowerBlack,lowerBrown,lowerBlue,lowerGreen,lowerGray,lowerOrange,lowerPink,lowerPurple,lowerRed,lowerWhite,lowerYellow\n')
        for i in range(len(res_list)):
            line = [res_list[i][0],res_list[i][1][0],res_list[i][1][1],res_list[i][1][2],
                    res_list[i][1][3],res_list[i][1][4],res_list[i][1][5],
                    res_list[i][1][6],
                    ','.join([str(x) if x>0 else '' for x in res_list[i][1][7:18]]),
                    ','.join([str(x) if x>0 else '' for x in res_list[i][1][18:]])]
            line = ','.join(line)
            line = line.replace('1.0','1')
            f.write(line+"\n")



if __name__ == '__main__':
    main(cfg)