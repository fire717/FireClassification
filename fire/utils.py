
import os
import torch
import random
import numpy as np


VERSION = "1.1"

def setRandomSeed(seed=42):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    setRandomSeed(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def printDash(num = 50):
    print(''.join(['-']*num))


def initFire(cfg):

    if cfg["cfg_verbose"]:
        printDash()
        print(cfg)
        printDash()

    print("[INFO] Fire verison: "+VERSION)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    setRandomSeed(cfg['random_seed'])

    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])



def npSoftmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def firelog(mode='i',text=''):
    if mode=='i':
        print("[INFO] ",text)


def delete_all_pycache_folders(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for dirname in dirnames:
            if dirname == "__pycache__":
                #os.rmdir(os.path.join(dirpath, dirname))
                os.system("rm -rf %s" % os.path.join(dirpath, dirname))