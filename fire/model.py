import torch
import torch.nn as nn

import pretrainedmodels
from efficientnet_pytorch import EfficientNet

from fire.models.mobilenetv3 import MobileNetV3
import torchvision

class FireModel(nn.Module):
    def __init__(self, cfg):
        super(FireModel, self).__init__()

        self.cfg = cfg
        

        self.pretrainedModel()
        
        self.changeModelStructure()
        

    
    def forward(self, img):        

        if self.cfg['model_name'] in ['mobilenetv2']:

            out = self.features(img)

            out = nn.functional.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)
            #nn.AdaptiveAvgPool2d(1)
            out = self.classifier(out)

        elif self.cfg['model_name'] in ['mobilenetv3']:

            out = self.features(img)

            out = out.mean(3).mean(2)        #best 99919
            #out = out.view(out.size(0), -1) #best 9990
            # print(out.shape)
            # b
            out = self.classifier(out)


        elif "efficientnet" in self.cfg['model_name']:
            out = self.pretrain_model(img)

        elif 'resnext' in self.cfg['model_name'] or \
                'xception' in self.cfg['model_name']:
            out = self.pretrain_model(img)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.last_linear(out)

        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])

        return out



    def changeModelStructure(self):
        ### Change model
        if self.cfg['model_name'] in ['mobilenetv2','mobilenetv3']:
            in_features = self.pretrain_model.classifier[1].in_features
            self.features = self.pretrain_model.features
            #self.avgpool = nn.AdaptiveAvgPool2d(1)
            #self.features[13] = nn.AdaptiveMaxPool2d(1)
            #print(self.features[13])
            # print(self.features)
            # b

            # self.classifier = self.pretrain_model.classifier
            # self.classifier[1] = nn.Linear(in_features, self.cfg['class_number'])

            self.classifier = nn.Sequential(
                nn.Dropout(p=self.cfg['dropout']),    # refer to paper section 6
                nn.Linear(in_features, self.cfg['class_number']),
            )


        elif "efficientnet" in self.cfg['model_name']:
            #self.pretrain_model._dropout = nn.Dropout(0.5)
            fc_features = self.pretrain_model._fc.in_features 
            self.pretrain_model._fc = nn.Linear(fc_features,  self.cfg['class_number'])
            # print(self.pretrain_model)
            # b
                                         # nn.ReLU(),  
                                         # nn.Dropout(0.25),
                                         # nn.Linear(512, 128), 
                                         # nn.ReLU(),  
                                         # nn.Dropout(0.50), 
                                         # nn.Linear(128,class_number))
            # print(list(self.pretrain_model.children())[:-3])
            # b
            # self.pretrain_models = nn.Sequential(*list(self.pretrain_model.children())[:-3])
            # self.last_linear = nn.Linear(fc_features, class_number) 

        # elif "RegNet" in self.cfg['model_name']:
        #     # print(model)
        #     # print(nn.Sequential(*list(model.children())[:-1]))
        #     # b
        #     fc_features = self.pretrain_model.head.fc.in_features 
        #     self.pretrain_model.head.fc = nn.Sequential(nn.Linear(fc_features, class_number))

        #     # for k,v in self.pretrain_model.named_parameters():
        #     #     print('{}: {}'.format(k, v.requires_grad))
        #     # b
        # elif "EN-B" in self.cfg['model_name']:
        #     # print(model)
        #     # print(nn.Sequential(*list(model.children())[:-1]))
        #     # b
        #     fc_features = self.pretrain_model.head.fc.in_features 
        #     self.pretrain_model.head.fc = nn.Sequential(nn.Linear(fc_features, class_number))
        
        elif 'resnext' in self.cfg['model_name'] or \
                'xception' in self.cfg['model_name']:
            #self.avgpool = nn.AdaptiveAvgPool2d(1)
            #print(self.pretrain_model)
            fc_features = self.pretrain_model.last_linear.in_features

            self.pretrain_model = nn.Sequential(*list(self.pretrain_model.children())[:-2])
            # self.dp_linear = nn.Linear(fc_features, 8) 
            # self.dp = nn.Dropout(0.50)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.last_linear = nn.Linear(fc_features, self.cfg['class_number']) 

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def pretrainedModel(self):


        ### Create model

        if self.cfg['model_name']=="mobilenetv2":
            #model.cpu()
            self.pretrain_model = torchvision.models.mobilenet_v2(pretrained=False, progress=True, width_mult=1.0)
            
            if self.cfg['pretrained']:
                state_dict = torch.load(self.cfg['pretrained'])
                self.pretrain_model.load_state_dict(state_dict, strict=True)



        elif self.cfg['model_name']=="mobilenetv3":
            #model.cpu()
            self.pretrain_model = MobileNetV3()
            if self.cfg['pretrained']:
                state_dict = torch.load(self.cfg['pretrained'])
                self.pretrain_model.load_state_dict(state_dict, strict=True)

            #in_features = self.pretrain_model.classifier[1].in_features
            #print(in_features)
            #self.pretrain_model.classifier[1] = nn.Linear(in_features, class_number)
            #print(self.pretrain_model)
            


        elif "efficientnet" in self.cfg['model_name']:
            #model = EfficientNet.from_name(model_name)
            self.pretrain_model = EfficientNet.from_name(self.cfg['model_name'].replace('adv-',''))
            if self.cfg['pretrained']:
                self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True) 

        
        elif 'resnext' in self.cfg['model_name'] or \
                'xception' in self.cfg['model_name']:
            #model_name = 'resnext50' # se_resnext50_32x4d xception
            self.pretrain_model = pretrainedmodels.__dict__[self.cfg['model_name']](num_classes=1000, pretrained=None)
            print(pretrainedmodels.pretrained_settings[self.cfg['model_name']])

            if self.cfg['pretrained']:
                if self.cfg['model_name']=="resnet50":
                    #model.cpu()
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True)
                    #fc_features = self.pretrain_model.last_linear.in_features 
                elif self.cfg['model_name']=="xception":
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True)
                    #fc_features = self.pretrain_model.last_linear.in_features 
                elif self.cfg['model_name'] == "se_resnext50_32x4d":
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True)
                    self.pretrain_model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                    #fc_features = self.pretrain_model.last_linear.in_features 
                elif self.cfg['model_name'] == "se_resnext101_32x4d":
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True)
                    self.pretrain_model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                    #fc_features = self.pretrain_model.last_linear.in_features 
                elif self.cfg['model_name'] == "resnext101_32x8d_wsl":
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True)
                    #fc_features = self.pretrain_model.fc.in_features 
                elif self.cfg['model_name'] == "resnext101_32x16d_wsl":
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True)
                    #fc_features = self.pretrain_model.fc.in_features 
                else:
                    raise Exception("[ERROR] Not load pretrained model!")
        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass

        # if 'RegNet' in self.cfg['model_name']:
        #     pass
        #     load_name = self.cfg['model_name']+'_dds_8gpu.pyth'
        #     load_path = os.path.join("../model/dds_baselines/176245422", load_name)
        #     # # model.load_state_dict(torch.load(load_path),strict=True) 
        #     checkpoint = torch.load(load_path, map_location="cpu")

        #     self.pretrain_model.load_state_dict(checkpoint["model_state"],strict=True)

        # elif 'EN-B' in self.cfg['model_name']:
        #     pass
        #     load_name = self.cfg['model_name']+'_dds_8gpu.pyth'
        #     load_path = os.path.join("../model/dds_baselines/161305098", load_name)
        #     # # model.load_state_dict(torch.load(load_path),strict=True) 
        #     checkpoint = torch.load(load_path, map_location="cpu")

        #     self.pretrain_model.load_state_dict(checkpoint["model_state"],strict=True)
            
        # else:
        #     raise Exception("[ERROR] Not load pretrained model!")

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])
