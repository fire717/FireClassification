import time
import gc
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import cv2

import torch.nn.functional as F

from fire.runnertools import getSchedu, getOptimizer, getLossFunc
from fire.runnertools import clipGradient
from fire.metrics import getF1
from fire.scheduler import GradualWarmupScheduler
from fire.utils import printDash,firelog,delete_all_pycache_folders




class FireRunner():
    def __init__(self, cfg, model):

        self.cfg = cfg

  
        if self.cfg['GPU_ID'] != '' :
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)


        self.scaler = torch.cuda.amp.GradScaler()
        ############################################################
        

        # loss
        self.loss_func = getLossFunc(self.device, cfg)
        

        
        # optimizer
        self.optimizer = getOptimizer(self.cfg['optimizer'], 
                                    self.model, 
                                    self.cfg['learning_rate'], 
                                    self.cfg['weight_decay'])


        # scheduler
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)
        
        if self.cfg['warmup_epoch']:
            self.scheduler = GradualWarmupScheduler(optimizer, 
                                                multiplier=1, 
                                                total_epoch=self.cfg['warmup_epoch'], 
                                                after_scheduler=self.scheduler)

        if self.cfg['show_heatmap']:
            self.extractor = ModelOutputs(self.model, self.model.features[12], ['0'])


    def freezeBeforeLinear(self, epoch, freeze_epochs = 2):
        if epoch<freeze_epochs:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif epoch==freeze_epochs:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = True
        #b


    def train(self, train_loader, val_loader):


        self.onTrainStart()

        for epoch in range(self.cfg['epochs']):

            self.freezeBeforeLinear(epoch, self.cfg['freeze_nonlinear_epoch'])

            self.onTrainStep(train_loader, epoch)

            #self.onTrainEpochEnd()

            self.onValidation(val_loader, epoch)



            if self.earlystop:
                break
        

        self.onTrainEnd()


    def predict(self, data_loader, return_raw=False):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data)[0]


                #print(output.shape)
                pred_score = nn.Softmax(dim=1)(output)
                #print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                for i in range(len(batch_pred_score)):
                    if return_raw:
                        res_dict[os.path.basename(img_names[i])] = pred_score[i].cpu().numpy()
                    else:
                        res_dict[os.path.basename(img_names[i])] = pred[i].item()
        return res_dict


    def cleanData(self, data_loader, target_label, move_dir):
        """
        input: data, move_path
        output: None

        """
        self.model.eval()
        
        #predict
        #res_list = []
        count = 0
        with torch.no_grad():
            #end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r",str(i)+"/"+str(data_loader.__len__()),end="",flush=True)

                inputs = inputs.cuda()

                output = self.model(inputs)
                output = nn.Softmax(dim=1)(output)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]

                    if np.argmax(output_one)!=target_label:
                        print(output_one, target_label,img_names[i])
                        img_name = os.path.basename(img_names[i])
                        os.rename(img_names[i], os.path.join(move_dir,img_name))
                        count += 1
        print("[INFO] Total: ",count)


    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0

        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                #print(target.shape)
                with torch.cuda.amp.autocast():
                    output = self.model(data)[0]

                # print(img_names)
                # print(output)
                pred_score = nn.Softmax(dim=1)(output)
                # print(pred_score)
                # b
                #print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                if self.cfg['use_distill']:
                    target = target.max(1, keepdim=True)[1] 
                correct += pred.eq(target.view_as(pred)).sum().item()


                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                batch_label_score = target.data.cpu().numpy().tolist()
                # print(batch_pred_score)
                # print(batch_label_score)
                # b
                pres.extend(batch_pred_score)
                labels.extend(batch_label_score)

        pres = np.array(pres)
        labels = np.array(labels)
        #print(pres.shape, labels.shape)

        acc =  correct / len(data_loader.dataset)


        print('[Info] acc: {:.3f}% \n'.format(100. * acc))

        if 'F1' in self.cfg['metrics']:
            precision, recall, f1_score = getF1(pres, labels)
            print('      precision: {:.5f}, recall: {:.5f}, f1_score: {:.5f}\n'.format(
                  precision, recall, f1_score))




    def make_save_dir(self):
        #exist_names = os.listdir(self.cfg['save_dir'])
        #print(os.walk(self.cfg['save_dir']))
        dirpath, dirnames, filenames = os.walk(self.cfg['save_dir']).__next__()
        exp_nums = []
        for name in dirnames:
            if name[:3]=='exp':
                try:
                    expid = int(name[3:])
                    exp_nums.append(expid)
                except:
                    continue
        new_id = 0
        if len(exp_nums)>0:
            new_id = max(exp_nums)+1
        exp_dir = os.path.join(self.cfg['save_dir'], 'exp'+str(new_id))

        firelog("i", "save to %s" % exp_dir)
        #if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        os.system("cp -r fire %s/" % exp_dir)
        delete_all_pycache_folders(exp_dir)
        os.system("cp config.py %s/" % exp_dir)
        return exp_dir

################

    def onTrainStart(self):
        
        self.last_best_value = 0
        self.last_best_dist = 0
        self.last_save_path = None

        self.earlystop = False
        self.best_epoch = 0

        # log
        self.log_time = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))

        self.exp_dir = self.make_save_dir()

    def onTrainStep(self,train_loader, epoch):
        
        self.model.train()
        correct = 0
        count = 0
        batch_time = 0
        total_loss = 0
        for batch_idx, (data, target, img_names) in enumerate(train_loader):

            one_batch_time_start = time.time()

            target = target.to(self.device)

            data = data.to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(data)
                #all_linear2_params = torch.cat([x.view(-1) for x in model.model_feature._fc.parameters()])
                #l2_regularization = 0.0003 * torch.norm(all_linear2_params, 2)
                loss = self.loss_func(output[0], target, self.cfg['sample_weights'],sample_weight_img_names=img_names)# + l2_regularization.item()    


            total_loss += loss.item()
            if self.cfg['clip_gradient']:
                clipGradient(self.optimizer, self.cfg['clip_gradient'])


            
            self.optimizer.zero_grad()#把梯度置零
            # loss.backward() #计算梯度
            # self.optimizer.step() #更新参数
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            ### train acc
            pred_score = nn.Softmax(dim=1)(output[0])
            pred = output[0].max(1, keepdim=True)[1] # get the index of the max log-probability
            if len(target.shape)>1:
                target = target.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)

            train_acc =  correct / count
            train_loss = total_loss/count
            #print(train_acc)
            one_batch_time = time.time() - one_batch_time_start
            batch_time+=one_batch_time
            # print(batch_time/(batch_idx+1), len(train_loader), batch_idx, 
            #     int(one_batch_time*(len(train_loader)-batch_idx)))
            eta = int((batch_time/(batch_idx+1))*(len(train_loader)-batch_idx-1))


            print_epoch = ''.join([' ']*(4-len(str(epoch+1))))+str(epoch+1)
            print_epoch_total = str(self.cfg['epochs'])+''.join([' ']*(4-len(str(self.cfg['epochs']))))

            log_interval = 10
            if batch_idx % log_interval== 0:
                print('\r',
                    '{}/{} [{}/{} ({:.0f}%)] - ETA: {}, loss: {:.4f}, acc: {:.4f}  LR: {:f}'.format(
                    print_epoch, print_epoch_total, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    datetime.timedelta(seconds=eta),
                    train_loss,train_acc,
                    self.optimizer.param_groups[0]["lr"]), 
                    end="",flush=True)




    def onTrainEnd(self):
        save_name = 'last.pt'
        self.last_save_path = os.path.join(self.exp_dir, save_name)
        self.modelSave(self.last_save_path)
        
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()


    def onValidation(self, val_loader, epoch):

        self.model.eval()
        self.val_loss = 0
        self.correct = 0


        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    self.val_loss += self.loss_func(output[0], target).item() # sum up batch loss

                #print(output.shape)
                pred_score = nn.Softmax(dim=1)(output[0])
                #print(pred_score.shape)
                pred = output[0].max(1, keepdim=True)[1] # get the index of the max log-probability
                if self.cfg['use_distill']:
                    target = target.max(1, keepdim=True)[1] 
                self.correct += pred.eq(target.view_as(pred)).sum().item()


                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                batch_label_score = target.data.cpu().numpy().tolist()
                pres.extend(batch_pred_score)
                labels.extend(batch_label_score)

        #print('\n',output[0],img_names[0])
        pres = np.array(pres)
        labels = np.array(labels)
        #print(pres.shape, labels.shape)

        self.val_loss /= len(val_loader.dataset)
        self.val_acc =  self.correct / len(val_loader.dataset)
        self.best_score = self.val_acc 

        if 'F1' in self.cfg['metrics']:
            #print(labels)
            precision, recall, f1_score = getF1(pres, labels)
            print(' \n           [VAL] loss: {:.5f}, acc: {:.3f}%, precision: {:.5f}, recall: {:.5f}, f1_score: {:.5f}\n'.format(
                self.val_loss, 100. * self.val_acc, precision, recall, f1_score))
            self.best_score = f1_score 

        else:
            print(' \n           [VAL] loss: {:.5f}, acc: {:.3f}% \n'.format(
                self.val_loss, 100. * self.val_acc))


        if self.cfg['warmup_epoch']:
            self.scheduler.step(epoch)
        else:
            if 'default' in self.cfg['scheduler']:
                self.scheduler.step(self.best_score)
            else:
                self.scheduler.step()


        self.checkpoint(epoch)
        self.earlyStop(epoch)

        


    def onTest(self):
        self.model.eval()
        
        #predict
        res_list = []
        with torch.no_grad():
            #end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)

                inputs = inputs.cuda()

                output = model(inputs)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]
                    output_one = np.argmax(output_one)

                    res_list.append(output_one)
        return res_list



    def earlyStop(self, epoch):
        ### earlystop
        if self.best_score>self.last_best_value:
            self.last_best_value = self.best_score
            self.last_best_dist = 0

        self.last_best_dist+=1
        if self.last_best_dist>self.cfg['early_stop_patient']:
            self.best_epoch = epoch-self.cfg['early_stop_patient']+1
            print("[INFO] Early Stop with patient %d , best is Epoch - %d :%f" % (self.cfg['early_stop_patient'],self.best_epoch,self.last_best_value))
            self.earlystop = True
        if  epoch+1==self.cfg['epochs']:
            self.best_epoch = epoch-self.last_best_dist+2
            print("[INFO] Finish trainging , best is Epoch - %d :%f" % (self.best_epoch,self.last_best_value))
            self.earlystop = True

    def checkpoint(self, epoch):
        
        if self.best_score<=self.last_best_value:
            pass
        else:
            save_name = 'best.pt'
            self.last_save_path = os.path.join(self.exp_dir, save_name)
            self.modelSave(self.last_save_path)




    def modelLoad(self,model_path, data_parallel = False):
        self.model.load_state_dict(torch.load(model_path), strict=True)
        
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def modelSave(self, save_name):
        torch.save(self.model.state_dict(), save_name)

    def toOnnx(self, save_name= "model.onnx"):
        dummy_input = torch.randn(1, 3, self.cfg['img_size'][0], self.cfg['img_size'][1]).to(self.device)

        torch.onnx.export(self.model, 
                        dummy_input, 
                        os.path.join(self.cfg['save_dir'],save_name), 
                        verbose=True)


