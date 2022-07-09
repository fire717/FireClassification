# @https://github.com/fire717/Fire

cfg = {
    ### Global Set
    "model_name": "mobilenetv3",  
    #mobilenetv3 adv-efficientnet-b2 se_resnext50_32x4d  xception resnext101_32x8d_wsl
    'GPU_ID': '0',
    "class_number": 10,

    "random_seed":42,
    "cfg_verbose":True,
    "num_workers":4,


    ### Train Setting
    'train_path':"./data/train",
    #../data/dataset/e_test
    #../data/dataset/d_trainval/v8/train/
    'label_type': 'DIR',# path or 'DIR' 
    'label_path': '',# if 'DIR' quale  train_path
    'val_path':"./data/val",
    'pretrained':'', #path or ''
    #pretrained/mobilenetv3_small_67.4.pth.tar
    #pretrained/mobilenet_v2-b0353104.pth
    #pretrained/se_resnext50_32x4d-a260b3a4.pth
    'log_interval':10,  
    'try_to_train_items': 200,   # 0 means all
    'save_best_only': True,  #only save model if better than before
    'save_one_only':True,    #only save one best model (will del model before)
    "save_dir": "output/",
    'pin_memory': True,
    'metrics': ['acc'], # default is acc,  can add F1  ...
    "loss": 'CE', # default or '' means CE, can other be Focalloss-1, BCE...

    'show_heatmap':False,


    ### Train Hyperparameters
    "img_size": [224, 224], # [h, w]
    'learning_rate':0.001,
    'batch_size':64,
    'epochs':100,
    'optimizer':'SGD',  #Adam  SGD AdaBelief Ranger
    'scheduler':'default-0.1-3', #default  SGDR-5-2  CVPR   step-4-0.8

    'warmup_epoch':0, # 
    'weight_decay' : 0,#0.0001,
    "k_flod":5,
    'start_fold':0,
    'early_stop_patient':7,

    'use_distill':0,
    'label_smooth':0,
    'checkpoint':None,
    'class_weight': None,#s[1.4, 0.78], # None [1, 1]
    'clip_gradient': 0,#1,       # 0
    'freeze_nonlinear_epoch':0,

    'dropout':0.5, #before last_linear

    'mixup':False,
    'sample_weights':None,


    ### Test
    'model_path':'output/mobilenetv3_e11_0.93300.pth',#test model

    'eval_path':"./data/test",#test with label,get test acc
    'test_path':"./data/test",#test without label, just show img result
    'use_TTA':0,
    'test_batch_size': 1,
    

}
