import torch
import torch.nn as nn
from fire.models.mobilenetv3 import MobileNetV3


class FireModel(nn.Module):
    def __init__(self, cfg):
        super(FireModel, self).__init__()

        self.cfg = cfg
        

        ### load model
        if self.cfg['model_name']=="mobilenetv3":
            #model.cpu()
            self.pretrain_model = MobileNetV3()
            if self.cfg['pretrained']:
                state_dict = torch.load(self.cfg['pretrained'])
                self.pretrain_model.load_state_dict(state_dict, strict=True)

            in_features = self.pretrain_model.classifier[1].in_features
            #print(in_features)
            #self.pretrain_model.classifier[1] = nn.Linear(in_features, class_number)
            #print(self.pretrain_model)
            

            self.features = self.pretrain_model.features
            #self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.8),    # refer to paper section 6
                nn.Linear(in_features, self.cfg['class_number']),
            )



        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass


        else:
            raise Exception("[ERROR] Unknown model_name!")
        

    def forward(self, img):        

        if self.cfg['model_name']=="mobilenetv3":
            out = self.features(img)
            # print(out.shape)
            out = out.mean(3).mean(2)        #best 99919
            #out = out.view(out.size(0), -1) #best 9990
            # print(out.shape)
            # b
            out = self.classifier(out)

        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass

        else:
            raise Exception("[ERROR] Unknown model_name!")

        return out