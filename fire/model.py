import torch
import torch.nn as nn

import pretrainedmodels
from efficientnet_pytorch import EfficientNet

from fire.models.mobilenetv3 import MobileNetV3


class FireModel(nn.Module):
    def __init__(self, cfg):
        super(FireModel, self).__init__()

        self.cfg = cfg
        

        self.pretrainedModel()
        
        self.changeModelStructure()
        

    
    def forward(self, img):        

        if self.cfg['model_name']=="mobilenetv3":

            out = self.features(img)

            out = out.mean(3).mean(2)        #best 99919
            #out = out.view(out.size(0), -1) #best 9990
            # print(out.shape)
            # b
            out = self.classifier(out)

        elif self.cfg['model_name']=="xception":
            out = self.model_feature(img)
            out = self.avgpool(out)

        elif "efficientnet" in self.cfg['model_name']:
            out = self.pretrain_model(img)

        elif 'resnext' in self.cfg['model_name'] or \
                'xception' in self.cfg['model_name']:
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
        if self.cfg['model_name'] == "mobilenetv3":
            in_features = self.pretrain_model.classifier[1].in_features
            self.features = self.pretrain_model.features
            #self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.8),    # refer to paper section 6
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
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.pretrain_model = nn.Sequential(*list(model.children())[:-1])
            
            # self.dp_linear = nn.Linear(fc_features, 8) 
            # self.dp = nn.Dropout(0.50)
            self.last_linear = nn.Linear(fc_features, class_number) 

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def pretrainedModel(self):
        ### Create model
        if self.cfg['model_name']=="mobilenetv3":
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
                self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=False) 

        
        elif 'resnext' in self.cfg['model_name'] or \
                'xception' in self.cfg['model_name']:
            #model_name = 'resnext50' # se_resnext50_32x4d xception
            self.pretrain_model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            print(pretrainedmodels.pretrained_settings[model_name])

            if self.cfg['model_name']=="resnet50":
                #model.cpu()
                self.pretrain_model.load_state_dict(torch.load("../pretrained/resnet50-19c8e357.pth"),strict=False)
                fc_features = self.pretrain_model.last_linear.in_features 
            elif self.cfg['model_name']=="xception":
                self.pretrain_model.load_state_dict(torch.load("../pretrained/xception-43020ad28.pth"),strict=False)
                fc_features = self.pretrain_model.last_linear.in_features 
            elif self.cfg['model_name'] == "se_resnext50_32x4d":
                self.pretrain_model.load_state_dict(torch.load("../pretrained/se_resnext50_32x4d-a260b3a4.pth"),strict=False)
                self.pretrain_model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                fc_features = self.pretrain_model.last_linear.in_features 
            elif self.cfg['model_name'] == "se_resnext101_32x4d":
                self.pretrain_model.load_state_dict(torch.load("../pretrained/se_resnext101_32x4d-3b2fe3d8.pth"),strict=False)
                self.pretrain_model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                fc_features = self.pretrain_model.last_linear.in_features 
            elif self.cfg['model_name'] == "resnext101_32x8d_wsl":
                self.pretrain_model.load_state_dict(torch.load("../pretrained/ig_resnext101_32x8-c38310e5.pth"),strict=False)
                fc_features = self.pretrain_model.fc.in_features 
            elif self.cfg['model_name'] == "resnext101_32x16d_wsl":
                self.pretrain_model.load_state_dict(torch.load("../pretrained/ig_resnext101_32x16-c6f796b0.pth"),strict=False)
                fc_features = self.pretrain_model.fc.in_features 
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
