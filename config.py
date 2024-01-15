# @https://github.com/fire717/Fire

cfg = {
    ### Global Set
    "model_name": "resnet50",  
    'GPU_ID': '1',
    "class_number": 10,
    "class_names": [], #str in list or [] for DIR label

    "random_seed":42,
    "cfg_verbose":True,
    "num_workers":4,


    ### Train Setting
    'train_path':"./data/train",
    'val_path':"./data/val", #if '' mean use k_flod
    'pretrained':'', #path or ''


    'try_to_train_items': 0,   # 0 means all, or run part(200 e.g.) for bug test
    'save_best_only': True,  #only save model if better than before
    'save_one_only':True,    #only save one best model (will del model before)
    "save_dir": "output/",
    'metrics': ['acc'], # default acc,  can add F1  ...
    "loss": 'CE', 

    'show_heatmap':False,
    'show_data':False,


    ### Train Hyperparameters
    "img_size": [224, 224], # [h, w]
    'learning_rate':0.001,
    'batch_size':64,
    'epochs':100,
    'optimizer':'SGD',  #Adam  SGD AdaBelief Ranger
    'scheduler':'default-0.1-3', #default  SGDR-5-2    step-4-0.8

    'warmup_epoch':0, # 
    'weight_decay' : 0,#0.0001,
    "k_flod":5,
    'val_fold':0,
    'early_stop_patient':7,

    'use_distill':0,
    'label_smooth':0,
    # 'checkpoint':None,
    'class_weight': None,#s[1.4, 0.78], # None [1, 1]
    'clip_gradient': 0,#1,       # 0
    'freeze_nonlinear_epoch':0,


    'mixup':False,
    'cutmix':False,
    'sample_weights':None,


    ### Test
    'model_path':'output/exp2/best.pt',#test model

    'eval_path':"./data/test",#test with label,get eval result
    'test_path':"./data/test",#test without label, just show img result
    
    'TTA':False,
    'merge':False,
    'test_batch_size': 1,
    

}
