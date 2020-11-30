import time
import gc
import os
import datetime
import torch
import torch.nn as nn
import numpy as np

from fire.runnertools import getSchedu, getOptimizer, clipGradient, writeLogs
from fire.metrics import getF1
from fire.scheduler import GradualWarmupScheduler
from fire.utils import printDash


class FireRunner():
    def __init__(self, cfg, model):

        self.cfg = cfg

        



        if self.cfg['GPU_ID'] != '' :
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        ############################################################
        

    
        # loss
        if self.cfg['class_weight']:
            weight_loss = torch.DoubleTensor(self.cfg['class_weight']).to(self.device)
        else:
            weight_loss = torch.DoubleTensor([1]*self.cfg['class_number']).to(self.device)
        self.loss_func = torch.nn.CrossEntropyLoss(weight=weight_loss).to(self.device)
        #self.loss_func = CrossEntropyLossOneHot().to(self.device)
        
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





    def train(self, train_loader, val_loader):


        self.onTrainStart()

        for epoch in range(self.cfg['epochs']):

            self.onTrainStep(train_loader, epoch)

            self.onTrainEpochEnd()

            self.onValidation(val_loader, epoch)



            if self.earlystop:
                break
        

        self.onTrainEnd()



    def predict(self):

        pass

    def evaluate(self):
        pass


    def onTrainStart(self):
        

        self.early_stop_value = 0
        self.early_stop_dist = 0
        self.last_save_path = None

        self.earlystop = False
        self.best_epoch = 0


    def onTrainStep(self,train_loader, epoch):
        
        self.model.train()
        correct = 0
        count = 0
        batch_time = 0
        for batch_idx, (data, target, img_names) in enumerate(train_loader):
            one_batch_time_start = time.time()
            data, target = data.to(self.device), target.to(self.device)


            output = self.model(data).double()

            #all_linear2_params = torch.cat([x.view(-1) for x in model.model_feature._fc.parameters()])
            #l2_regularization = 0.0003 * torch.norm(all_linear2_params, 2)

            loss = self.loss_func(output, target)# + l2_regularization.item()
            loss.backward() #计算梯度


            if self.cfg['clip_gradient']:
                clipGradient(self.optimizer, self.cfg['clip_gradient'])


            self.optimizer.step() #更新参数
            self.optimizer.zero_grad()#把梯度置零

            ### train acc
            pred_score = nn.Softmax(dim=1)(output)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if self.cfg['use_distill'] or self.cfg['label_smooth']>0:
                target = target.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)

            train_acc =  correct / count
            #print(train_acc)
            one_batch_time = time.time() - one_batch_time_start
            batch_time+=one_batch_time
            # print(batch_time/(batch_idx+1), len(train_loader), batch_idx, 
            #     int(one_batch_time*(len(train_loader)-batch_idx)))
            eta = int((batch_time/(batch_idx+1))*(len(train_loader)-batch_idx-1))


            print_epoch = ''.join([' ']*(4-len(str(epoch+1))))+str(epoch+1)
            print_epoch_total = str(self.cfg['epochs'])+''.join([' ']*(4-len(str(self.cfg['epochs']))))
            if batch_idx % self.cfg['log_interval'] == 0:
                print('\r',
                    '{}/{} [{}/{} ({:.0f}%)] - ETA: {}, loss: {:.4f}, acc: {:.4f} '.format(
                    print_epoch, print_epoch_total, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    datetime.timedelta(seconds=eta),
                    loss.item(),train_acc), 
                    end="",flush=True)


        

    def onTrainEpochEnd(self):
        print(" LR:", '{:.6f}'.format(self.optimizer.param_groups[0]["lr"]), end="")


    def onTrainEnd(self):

        writeLogs(self.cfg,self.best_epoch,self.early_stop_value)

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
                #print(target.shape)
                output = self.model(data).double()


                self.val_loss += self.loss_func(output, target).item() # sum up batch loss

                #print(output.shape)
                pred_score = nn.Softmax(dim=1)(output)
                #print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                if self.cfg['use_distill'] or self.cfg['label_smooth']>0:
                    target = target.max(1, keepdim=True)[1] 
                self.correct += pred.eq(target.view_as(pred)).sum().item()


                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                batch_label_score = target.data.cpu().numpy().tolist()
                pres.extend(batch_pred_score)
                labels.extend(batch_label_score)

        pres = np.array(pres)
        labels = np.array(labels)
        #print(pres.shape, labels.shape)

        self.val_loss /= len(val_loader.dataset)
        self.val_acc =  self.correct / len(val_loader.dataset)

        if 'F1' in self.cfg['metrics']:
            precision, recall, f1_score = getF1(pres, labels)
            print(' \n           [VAL] loss: {:.5f}, acc: {:.3f}%, precision: {:.5f}, recall: {:.5f}, f1_score: {:.5f}\n'.format(
                self.val_loss, 100. * self.val_acc, precision, recall, f1_score))

        else:
            print(' \n           [VAL] loss: {:.5f}, acc: {:.3f}% \n'.format(
                self.val_loss, 100. * self.val_acc))




        

        if self.cfg['warmup_epoch']:
            self.scheduler.step(epoch)
        else:
            if self.cfg['scheduler']=='default':
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()


        #print("---")
        #print(val_acc, early_stop_value, early_stop_dist)
        if self.val_acc>self.early_stop_value:
            self.early_stop_value = self.val_acc
            self.early_stop_dist = 0
                
            if self.cfg['save_best_only']:
                if self.last_save_path is not None and os.path.exists(self.last_save_path):
                    os.remove(self.last_save_path)
            save_name = '%s_e%d_%.5f.pth' % (self.cfg['model_name'],epoch+1,self.val_acc)
            self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
            torch.save(self.model.state_dict(), self.last_save_path)

        

        

        self.early_stop_dist+=1
        if self.early_stop_dist>self.cfg['early_stop_patient']:
            self.best_epoch = epoch-self.cfg['early_stop_patient']+1
            print("[INFO] Early Stop with patient %d , best is Epoch - %d :%f" % (self.cfg['early_stop_patient'],self.best_epoch,self.early_stop_value))
            self.earlystop = True
        if  epoch+1==self.cfg['epochs']:
            self.best_epoch = epoch-self.early_stop_dist+2
            print("[INFO] Finish trainging , best is Epoch - %d :%f" % (self.best_epoch,self.early_stop_value))
            self.earlystop = True




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
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]
                    output_one = np.argmax(output_one)

                    
                    if output_one!=target_label:
                        #print(output_one, target_label,img_names[i])
                        img_name = os.path.basename(img_names[i])
                        os.rename(img_names[i], os.path.join(move_dir,img_name))
                        count += 1
        print("[INFO] Total: ",count)




    def modelLoad(self,model_path):
        self.model.load_state_dict(torch.load(model_path))
        
        self.model = torch.nn.DataParallel(self.model)

    def modelSave(self):
        pass
