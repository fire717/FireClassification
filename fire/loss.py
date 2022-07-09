
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###########################  loss

def labelSmooth(one_hot, label_smooth):
    return one_hot*(1-label_smooth)+label_smooth/one_hot.shape[1]


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))


class CrossEntropyLossV2(nn.Module):
    def __init__(self, label_smooth=0, weight=None):
        super().__init__()
        self.weight = weight 
        self.label_smooth = label_smooth
        self.epsilon = 1e-7
        
    def forward(self, x, y, label_smooth=0, gamma=0, sample_weights=None, sample_weight_img_names=None):

        #one_hot_label = F.one_hot(y, x.shape[1])
        one_hot_label = y
        if label_smooth:
            one_hot_label = labelSmooth(one_hot_label, label_smooth)

        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = F.softmax(x, 1)
        #print(y_softmax)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -one_hot_label * y_softmaxlog

        if sample_weights:
            loss = loss*torch.Tensor(sample_weights).to(loss.device)

        #focal loss gamma
        if gamma:
            loss = loss*((1-y_softmax)**gamma)

        loss = torch.mean(torch.sum(loss, -1))

        return 


class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth=0, weight=None):
        super().__init__()
        self.weight = weight 
        self.label_smooth = label_smooth
        self.epsilon = 1e-7
        
    def forward(self, x, y, sample_weights=0, sample_weight_img_names=None):

        one_hot_label = F.one_hot(y, x.shape[1])

        if self.label_smooth:
            one_hot_label = labelSmooth(one_hot_label, self.label_smooth)

        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = F.softmax(x, 1)
        #print(y_softmax)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -one_hot_label * y_softmaxlog

        loss = torch.mean(torch.sum(loss, -1))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, label_smooth=0, gamma = 0., weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight # means alpha
        self.epsilon = 1e-7
        self.label_smooth = label_smooth

        
    def forward(self, x, y, sample_weights=0, sample_weight_img_names=None):

        if len(y.shape) == 1:
            #
            one_hot_label = F.one_hot(y, x.shape[1])

            if self.label_smooth:
                one_hot_label = labelSmooth(one_hot_label, self.label_smooth)

            if sample_weights>0 and sample_weights is not None:
                #print(sample_weight_img_names)
                weigths = [sample_weights  if 'yxboard' in img_name  else 1 for img_name in sample_weight_img_names] 
                weigths = torch.DoubleTensor(weigths).reshape((len(weigths),1)).to(x.device)
                #print(weigths, weigths.shape)
                #print(one_hot_label, one_hot_label.shape)
                one_hot_label = one_hot_label*weigths
                #print(one_hot_label)
                #b
        else:
            one_hot_label = y


        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = F.softmax(x, 1)
        #print(y_softmax)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = torch.log(y_softmax)

        #print(y_softmaxlog)
        # original CE loss
        loss = -one_hot_label * y_softmaxlog
        #loss = 1 * torch.abs(one_hot_label-y_softmax)#my new CE..ok its L1...

        # print(one_hot_label)
        # print(y_softmax)
        # print(one_hot_label-y_softmax)
        # print(torch.abs(y-y_softmax))
        #print(loss)
        
        # gamma
        loss = loss*((torch.abs(one_hot_label-y_softmax))**self.gamma)
        # print(loss)

        # alpha
        if self.weight is not None:
            loss = self.weight*loss

        loss = torch.mean(torch.sum(loss, -1))
        return loss





if __name__ == '__main__':



    device = torch.device("cpu")

    #x = torch.randn(2,2)
    x = torch.tensor([[0.1,0.7,0.2]])
    y = torch.tensor([1])
    print(x)

    loss_func = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_func(x,y)
    print("loss1: ",loss)

    # loss_func = Focalloss().to(device)
    # loss = loss_func(x,y)
    # print("loss2: ",loss)
    

    weight_loss = torch.DoubleTensor([1,1,1]).to(device)
    loss_func = FocalLoss(gamma=0, weight=weight_loss).to(device)
    loss = loss_func(x,y)
    print("loss3: ",loss)
    

    # weight_loss = torch.DoubleTensor([2,1]).to(device)
    # loss_func = Focalloss(gamma=0.2, weight=weight_loss).to(device)
    # loss = loss_func(x,y)
    # print("loss4: ",loss)