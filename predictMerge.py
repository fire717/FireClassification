import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np





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
    runner1 = FireRunner(cfg, model)
    runner1.modelLoad('output/efficientnet-b6_e17_fold0_0.93368.pth')
    print("load model1, start running.")
    res_dict1 = runner1.predictRaw(test_loader)
    print(len(res_dict1))

    test_loader = data.getTestDataloader()
    runner2 = FireRunner(cfg, model)
    runner2.modelLoad('output/efficientnet-b6_e18_fold1_0.94537.pth')
    print("load model2, start running.")
    res_dict2 = runner2.predictRaw(test_loader)

    test_loader = data.getTestDataloader()
    runner3 = FireRunner(cfg, model)
    runner3.modelLoad('output/efficientnet-b6_e14_fold2_0.91967.pth')
    print("load model3, start running.")
    res_dict3 = runner3.predictRaw(test_loader)

    test_loader = data.getTestDataloader()
    runner4 = FireRunner(cfg, model)
    runner4.modelLoad('output/efficientnet-b6_e18_fold3_0.92239.pth')
    print("load model4, start running.")
    res_dict4 = runner4.predictRaw(test_loader)

    # test_loader = data.getTestDataloader()
    # runner5 = FireRunner(cfg, model)
    # runner5.modelLoad('output/efficientnet-b6_e17_fold0_0.93368.pth')
    # print("load model5, start running.")
    # res_dict5 = runner5.predictRaw(test_loader)


    res_dict = {}
    for k,v in res_dict1.items():
        #print(k,v)
        v1 =np.argmax(v+res_dict2[k]+res_dict3[k]+res_dict4[k])
        res_dict[k] = v1
    
    res_list = sorted(res_dict.items(), key = lambda kv: int(kv[0].split("_")[-1].split('.')[0]))
    print(len(res_list), res_list[0])

    # to csv
    # res_list_final = []
    # for res in res_list:
    #     res_list_final.append([res[0]]+res[1])
    # #res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['type'])
    # #res_df = res_df.reset_index().rename(columns={'index':'id'})
    # res_df = DataFrame(res_list_final, columns=['id','type','color','toward'])
    

    # res_df.to_csv(os.path.join(cfg['save_dir'], 'result.csv'), 
    #                             index=False,header=True)

    with open('result.csv', 'w', encoding='utf-8') as f:
        f.write('file,label\n')
        for i in range(len(res_list)):
            line = [res_list[i][0], str(res_list[i][1])]
            line = ','.join(line)
            f.write(line+"\n")



if __name__ == '__main__':
    main(cfg)