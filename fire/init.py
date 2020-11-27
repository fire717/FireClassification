import os

from fire._version import __version__
from fire.utils import setRandomSeed, printDash

def initFire(cfg):

    if cfg["cfg_verbose"]:
        printDash()
        print(cfg)
        printDash()

    print("[INFO] Fire verison: "+__version__)


    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    setRandomSeed(cfg['random_seed'])



    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])