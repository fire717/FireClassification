from fire.models.mobileformer import MobileFormer
import torch


def mobile_former_custom(args, pre_train=False, state_dir=None):
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    return model

def mobile_former_508(pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[144, 120], [240, 216], [432, 512, 768, 1056], [1056, 1440, 1440]],
        'out_channels' : [[40, 40], [72, 72], [128, 128, 176, 176], [240, 240, 240]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'stem_out_channel' : 24, 
        'bneck_exp' : 48, 'bneck_out' : 24,
        'project_demension' : 1440, 'fc_demension' : 1920
    }
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    return model

def mobile_former_294(pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[96, 96], [144, 192], [288, 384, 576, 768], [768, 1152, 1152]],
        'out_channels' : [[24, 24], [48, 48], [96, 96, 128, 128], [192, 192, 192]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'stem_out_channel' : 16, 
        'bneck_exp' : 32, 'bneck_out' : 16,
        'project_demension' : 1152, 'fc_demension' : 1920
    }
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    return model

def mobile_former_214(pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[72, 60], [120, 160], [240, 320, 480, 672], [672, 960, 960]],
        'out_channels' : [[20, 20], [40, 40], [80, 80, 112, 112], [160, 160, 160]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'stem_out_channel' : 12, 
        'bneck_exp' : 24, 'bneck_out' : 12,
        'project_demension' : 960, 'fc_demension' : 1600
    }
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    return model

def mobile_former_151(pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[72, 48], [96, 96], [192, 256, 384, 528], [528, 768, 768]],
        'out_channels' : [[16, 16], [32, 32], [64, 64, 88, 88], [128, 128, 128]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'stem_out_channel' : 12, 
        'bneck_exp' : 24, 'bneck_out' : 12,
        'project_demension' : 768, 'fc_demension' : 1280
    }
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    return model

def mobile_former_96(pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[72], [96, 96], [192, 256, 384], [528, 768]],
        'out_channels' : [[16], [32, 32], [64, 64, 88], [128, 128]],
        'num_token' : 4, 'd_model' : 128, 
        'in_channel' : 3, 'stem_out_channel' : 12, 
        'bneck_exp' : 24, 'bneck_out' : 12,
        'project_demension' : 768, 'fc_demension' : 1280
    }
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    return model


def mobile_former(num):
    if num==96:
        return mobile_former_96()
    elif num==151:
        return mobile_former_151()
    elif num==214:
        return mobile_former_214()
    elif num==294:
        return mobile_former_294()
    elif num==508:
        return mobile_former_508()
    else:
        raise Exception("Unknow mobile_former num: ", num)