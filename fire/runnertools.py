import os
import time
import torch
import torch.optim as optim
from fire.ranger import Ranger 
from fire.loss import FocalLoss, CrossEntropyLoss

def getSchedu(schedu, optimizer):
    if schedu=='default':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1)
    elif 'step' in schedu:
        step_size = int(schedu.strip().split('-')[1])
        gamma = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif 'SGDR' in schedu: 
        T_0 = int(schedu.strip().split('-')[1])
        T_mult = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=T_0, 
                                                            T_mult=T_mult)
    return scheduler

def getOptimizer(optims, model, learning_rate, weight_decay):
    if optims=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optims=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optims=='AdaBelief':
        optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-12, betas=(0.9,0.999))
    elif optims=='Ranger':
        optimizer = Ranger(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def getLossFunc(device, cfg):
    # loss

    if cfg['class_weight']:
        loss_weight = torch.DoubleTensor(cfg['class_weight']).to(device)
    else:
        loss_weight = torch.DoubleTensor([1]*cfg['class_number']).to(device)
    


    if 'Focalloss' in cfg['loss']:
        gamma = float(cfg['loss'].strip().split('-')[1])
        loss_func = FocalLoss(label_smooth=cfg['label_smooth'],
                            gamma=gamma,
                            weight=loss_weight).to(device)


    else:
        # loss_func = torch.nn.CrossEntropyLoss(weight=loss_weight).to(device)
        loss_func = CrossEntropyLoss(label_smooth=cfg['label_smooth'],
                            weight=loss_weight).to(device)
            #self.loss_func = CrossEntropyLossOneHot().to(self.device)



    return loss_func




############### Tools

def clipGradient(optimizer, grad_clip=1):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def writeLogs(cfg, 
            best_epoch, 
            early_stop_value,
            log_time):
    # 可以自定义要保存的字段
    line_list= cfg['log_item']
    log_path = os.path.join(cfg['save_dir'], 'history.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            line = ','.join(['timestamps']+line_list+['best_epoch','best_value'])+"\n"
            f.write(line)

    with open(log_path, 'a', encoding='utf-8') as f:

        line_tmp = [log_time] + \
                    ['-'.join([str(v) for v in cfg[x]]) if isinstance(cfg[x],list) else cfg[x] for x in line_list] + \
                    [best_epoch, early_stop_value]
        line = ','.join([str(x) for x in line_tmp])+"\n"
        f.write(line)