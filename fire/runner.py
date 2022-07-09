import time
import gc
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import cv2

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from fire.runnertools import getSchedu, getOptimizer, getLossFunc
from fire.runnertools import clipGradient
from fire.metrics import getF1
from fire.scheduler import GradualWarmupScheduler
from fire.utils import printDash



class FeatureExtractor():
    #https://github.com/jacobgil/pytorch-grad-cam
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # print(self.model._modules.items())
        for name, module in self.model._modules.items():
            # print(name, module)
            x = module(x)
            if name in self.target_layers:
                # print(name, module, '111')
                x.register_hook(self.save_gradient)
                outputs += [x]
        # b
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        print(len(self.model.features._modules.items()))
        for name, module in self.model.features._modules.items():
            if module == self.feature_module:
                print(name, module)
                target_activations, x = self.feature_extractor(x)
            else:
                x = module(x)

            
        
        x = x.mean(3).mean(2)        #best 99919
        x = self.model.classifier(x)
        #bself.pretrain_model self.features self.classifier

        return target_activations, x




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


    def predictRaw(self, data_loader):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data).double()


                #print(output.shape)
                pred_score = nn.Softmax(dim=1)(output)
                #print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                for i in range(len(batch_pred_score)):
                    res_dict[os.path.basename(img_names[i])] = pred_score[i].cpu().numpy()

        # pres = np.array(pres)

        return res_dict

    def predict(self, data_loader):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data).double()


                #print(output.shape)
                pred_score = nn.Softmax(dim=1)(output)
                #print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                for i in range(len(batch_pred_score)):
                    res_dict[os.path.basename(img_names[i])] = pred[i].item()

        # pres = np.array(pres)

        return res_dict

    def heatmap(self, data_loader, save_dir, count):
        print("[DEBUG] Not finish")
        self.model.eval()
        c = 0


        res_dict = {}
        # with torch.no_grad():

        # x=torch.randn(3)
        # x=torch.autograd.Variable(x,requires_grad=True)#生成变量
        # print(x)#输出x的值
        # y=x*2
        # y = y.requires_grad_(True) 
        # y.backward(torch.FloatTensor([1,0.1,0.01]))#自动求导
        # print(x.grad)#求对x的梯度
        # print(x)
        # b

        pres = []
        labels = []
        show = 1
        for (data, img_names) in data_loader:
            data = data.to(self.device)


            features, output = self.extractor(data)
            print(np.array(features).shape, features[0].shape)
            print(output.shape)
            # output = self.model(data).double()
            if show:
                print('output: ', output, img_names)
                show = 0
            # pred = nn.Softmax(dim=1)(output)
            # print("pre: ", pred.cpu())


            ###  grad-CAM
            # print(self.model.features[12][0])
            # b
            # 利用onehot的形式锁定目标类别
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][0] = 1
            one_hot = torch.from_numpy(one_hot)
            # 获取目标类别的输出,该值带有梯度链接关系,可进行求导操作
            one_hot = torch.sum(one_hot.to(self.device) * output).requires_grad_(True) 
            self.model.zero_grad()
            one_hot.backward(retain_graph=True) # backward 求导
            # 获取对应特征层的梯度map
            #print(self.extractor.get_gradients().shape)
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
            # grads_val = self.model.features[12][0].grad
            print(grads_val.shape)
            target = features[-1] # 获取目标特征输出
            target = target.cpu().data.numpy()[0, :]

            weights = np.mean(grads_val, axis=(2, 3))[0, :] # 利用GAP操作, 获取特征权重
            cam = np.zeros(target.shape[1:], dtype=np.float32)
            # relu操作,去除负值, 并缩放到原图尺寸
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, data.shape[2:])
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            print(cam.shape)
            print(cam[0][:10])

            origin_img = cv2.imread(img_names[0])

            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, "mask1_"+os.path.basename(img_names[0])), 
                            heatmap)
            # 0.4 here is a heatmap intensity factor
            superimposed_img = heatmap * 0.4 + origin_img
            # Save the image to disk
            cv2.imwrite(os.path.join(save_dir, os.path.basename(img_names[0])), 
                            superimposed_img)
            

            # print(self.model.features[:])
            # b
            # res
            
            # with torch.no_grad():
            #     print('-------')
            #     features = self.model.features[:13](data)
            #     # out = self.model.features[13:](features)
            #     # out = out.mean(3).mean(2)        #best 99919
            #     # # print(out.shape)
            #     # # b
            #     # out = self.model.classifier(out)
                
            #     # print('output2', out)
            #     weights = nn.functional.adaptive_avg_pool2d(features,(1,1))
            #     weights_value = weights.cpu().numpy()
            #     # weights_value = np.clip(weights_value,-1e-7, 1)
            #     # weights_value = np.reshape(weights_value, (weights_value.shape[1],))
            #     print(weights_value.shape)
            #     #print(weights_value[0])

            #     features_value = features.cpu().numpy()
            #     print(weights_value.shape, features_value.shape)
            #     print(features_value[0][0][0])
            #     heatmap = weights_value*features_value

            #     print(heatmap.shape, heatmap[0][0][0])
            #     heatmap = np.mean(heatmap, axis = 1)
            #     print(heatmap.shape, heatmap[0][0])
            #     #b

            #     heatmap = np.maximum(heatmap, 0)
            #     heatmap = heatmap[0]
                
            #     #heatmap = np.reshape(heatmap, (heatmap.shape[1], heatmap.shape[2]))
            #     #print(heatmap.shape )

            #     origin_img = cv2.imread(img_names[0])
            #     #print(origin_img.shape)
            #     # We resize the heatmap to have the same size as the original image
            #     heatmap = cv2.resize(heatmap, (origin_img.shape[1], origin_img.shape[0]))
            #     # cv2.imwrite(os.path.join(save_dir, "mask0_"+os.path.basename(img_names[0])), 
            #     #                 heatmap)
            #     # We convert the heatmap to RGB
            #     heatmap = np.uint8(255 * heatmap)
            #     # cv2.imwrite(os.path.join(save_dir, "mask1_"+os.path.basename(img_names[0])), 
            #     #                 heatmap)
            #     # We apply the heatmap to the original image
            #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            #     cv2.imwrite(os.path.join(save_dir, "mask2_"+os.path.basename(img_names[0])), 
            #                     heatmap)
            #     # 0.4 here is a heatmap intensity factor
            #     #superimposed_img = heatmap * 0.4 + origin_img

            #     # Save the image to disk
            #     # cv2.imwrite(os.path.join(save_dir, os.path.basename(img_names[0])), 
            #     #                 superimposed_img)

            c+=1
            if c==count:
                return

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
                    output = self.model(data).double()

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

################

    def onTrainStart(self):
        

        self.early_stop_value = 0
        self.early_stop_dist = 0
        self.last_save_path = None

        self.earlystop = False
        self.best_epoch = 0

        # log
        self.log_time = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))


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
        if self.best_score>self.early_stop_value:
            self.early_stop_value = self.best_score
            self.early_stop_dist = 0

        self.early_stop_dist+=1
        if self.early_stop_dist>self.cfg['early_stop_patient']:
            self.best_epoch = epoch-self.cfg['early_stop_patient']+1
            print("[INFO] Early Stop with patient %d , best is Epoch - %d :%f" % (self.cfg['early_stop_patient'],self.best_epoch,self.early_stop_value))
            self.earlystop = True
        if  epoch+1==self.cfg['epochs']:
            self.best_epoch = epoch-self.early_stop_dist+2
            print("[INFO] Finish trainging , best is Epoch - %d :%f" % (self.best_epoch,self.early_stop_value))
            self.earlystop = True

    def checkpoint(self, epoch):
        
        if self.best_score<=self.early_stop_value:
            if self.cfg['save_best_only']:
                pass
            else:
                save_name = '%s_e%d_%.5f.pth' % (self.cfg['model_name'],epoch+1,self.best_score)
                self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
                self.modelSave(self.last_save_path)
        else:
            if self.cfg['save_one_only']:
                if self.last_save_path is not None and os.path.exists(self.last_save_path):
                    os.remove(self.last_save_path)
            save_name = '%s_e%d_%.5f.pth' % (self.cfg['model_name'],epoch+1,self.best_score)
            self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
            self.modelSave(self.last_save_path)




    def modelLoad(self,model_path, data_parallel = True):
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


