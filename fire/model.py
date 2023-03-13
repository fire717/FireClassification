import torch
import torch.nn as nn

import pretrainedmodels

from fire.models.mobilenetv3 import MobileNetV3
from fire.models.micronet import MicroNet

from fire.models.myefficientnet_pytorch import EfficientNet
from fire.models.convnext import convnext_tiny,convnext_small,convnext_base,convnext_large
from fire.models.swin import build_model,get_config


import torchvision

class FireModel(nn.Module):
    def __init__(self, cfg):
        super(FireModel, self).__init__()

        self.cfg = cfg
        

        self.pretrainedModel()
        
        self.changeModelStructure()
        


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


        elif "shufflenetv2" in self.cfg['model_name']:
            self.pretrain_model = torchvision.models.shufflenet_v2_x1_0()
            if self.cfg['pretrained']:
                self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True)


        elif "micronet" in self.cfg['model_name']:
            #model.cpu()
            m = self.cfg['model_name'].strip().split('-')[-1]
            self.pretrain_model = MicroNet(self.cfg['class_number'],m)
            if self.cfg['pretrained']:
                state_dict = torch.load(self.cfg['pretrained'])
                self.pretrain_model.load_state_dict(state_dict, strict=True)


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

        elif "swin" in self.cfg['model_name']:
            if 'base' in self.cfg['model_name']:
                cfg = "fire/models/swin/configs/swin_base_patch4_window12_384_finetune.yaml"
                config = get_config(cfg)

                self.pretrain_model = build_model(config)

                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 

            elif 'small' in self.cfg['model_name']:
                cfg = "fire/models/swin/configs/swin_small_patch4_window7_224.yaml"
                config = get_config(cfg)

                self.pretrain_model = build_model(config)

                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 

            elif 'large' in self.cfg['model_name']:
                cfg = "fire/models/swin/configs/swin_large_patch4_window12_384_22kto1k_finetune.yaml"
                config = get_config(cfg)

                self.pretrain_model = build_model(config)

                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 

        elif "convnext" in self.cfg['model_name']:
            if "base" in self.cfg['model_name']:
                self.pretrain_model = convnext_base()
                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 
                # print(self.pretrain_model)
                # b
            elif "tiny" in self.cfg['model_name']:
                self.pretrain_model = convnext_tiny()
                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 
                # print(self.pretrain_model)
                # b
            elif "small" in self.cfg['model_name']:
                self.pretrain_model = convnext_small()
                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 
            elif "large" in self.cfg['model_name']:
                self.pretrain_model = convnext_large()
                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 
        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass


        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def changeModelStructure(self):
        ### Change model
        if 'mobilenetv2' in self.cfg['model_name']:

            in_features = self.pretrain_model.classifier[1].in_features
            self.features = self.pretrain_model.features

            self.head1 = nn.Sequential(
                nn.Dropout(p=0.2),    # refer to paper section 6
                nn.Linear(in_features, self.cfg['class_number']),
            )

        elif "mobilenetv3" in self.cfg['model_name']:
            # self.backbone =  self.pretrain_model
            self.backbone = nn.Sequential(*list(self.pretrain_model.children())[:-1])
            # print(self.backbone)
            # b
            num_features = 1280
            self.head1 = nn.Sequential(
                         # nn.Linear(num_features, 64),
                         nn.Dropout(0.8),
                         # nn.AdaptiveAvgPool2d(1),
                         nn.Linear(num_features, self.cfg['class_number']))

        elif "shufflenetv2" in self.cfg['model_name']:
            # self.backbone =  self.pretrain_model
            self.backbone = nn.Sequential(*list(self.pretrain_model.children())[:-1])
            # print(self.backbone)
            # b
            num_features = 1024
            # self.head1 = nn.Sequential(
            #              # nn.Linear(num_features, 64),
            #              nn.Dropout(0.8),
            #              # nn.AdaptiveAvgPool2d(1),
            #              nn.Linear(num_features,4))
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head1 = nn.Linear(num_features,self.cfg['class_number'])

        elif "micronet" in self.cfg['model_name']:
            self.features = self.pretrain_model


        elif "efficientnet" in self.cfg['model_name']:
            #self.pretrain_model._dropout = nn.Dropout(0.5)
            fc_features = self.pretrain_model._fc.in_features 
            self.pretrain_model._fc = nn.Linear(fc_features,  self.cfg['class_number'])


        elif "convnext" in self.cfg['model_name']:

            self.backbone =  self.pretrain_model
            #print(self.backbone)
            num_features = 1024
            if "large" in self.cfg['model_name']:
                num_features = 1536
            elif "tiny" in self.cfg['model_name']:
                num_features = 768
            elif "small" in self.cfg['model_name']:
                num_features = 768

            self.head1 = nn.Sequential(
                         # nn.Dropout(0.5),
                         nn.Linear(num_features,self.cfg['class_number']))


        elif "swin" in self.cfg['model_name']:
            self.backbone =  self.pretrain_model
            #print(self.backbone)
            num_features = self.backbone.norm.weight.size()[0]

            self.head1 = nn.Sequential(
                         # nn.Dropout(0.5),
                         nn.Linear(num_features,self.cfg['class_number']))


        elif 'resnext' in self.cfg['model_name'] or \
                'xception' in self.cfg['model_name']:
            #self.avgpool = nn.AdaptiveAvgPool2d(1)
            #print(self.pretrain_model)
            fc_features = self.pretrain_model.last_linear.in_features

            self.pretrain_model = nn.Sequential(*list(self.pretrain_model.children())[:-2])
            # self.dp_linear = nn.Linear(fc_features, 8) 
            # self.dp = nn.Dropout(0.50)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head1 = nn.Linear(fc_features, self.cfg['class_number']) 

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def forward(self, img):        

        if self.cfg['model_name'] in ['mobilenetv2']:

            out = self.features(img)

            out = nn.functional.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)
            #nn.AdaptiveAvgPool2d(1)
            out = self.head1(out)
            out = [out1] 

        elif "shuffle" in self.cfg['model_name']:
            out = self.backbone(img)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out1 = self.head1(out)

            out = [out1]

        elif "mobilenetv3" in self.cfg['model_name']:
            out = self.backbone(img)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            out = out.mean(3).mean(2)  
            out1 = self.head1(out)

            out = [out1]


        elif "micronet" in self.cfg['model_name']:
            out = self.features(img)
            out = [out1]

        elif "efficientnet" in self.cfg['model_name']:
            out = self.backbone(img)
            out = out.view(out.size(0), -1)
            out1 = self.head1(out)

            out = [out1]

        elif "swin" in self.cfg['model_name']:
            out = self.backbone(img)
            out = out.view(out.size(0), -1)
            out1 = self.head1(out)

            out = [out1]

        elif "convnext" in self.cfg['model_name']:

            out = self.backbone(img)
            out = out.view(out.size(0), -1)
            #print(out.shape)
            out1 = self.head1(out)

            out = [out1]

        elif 'resnext' in self.cfg['model_name'] or \
                'xception' in self.cfg['model_name']:
            out = self.pretrain_model(img)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.head1(out)
            out = [out1]
        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])

        return out


