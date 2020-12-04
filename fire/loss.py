
import torch
import torch.nn as nn
import torch.nn.functional as F

###########################  loss

class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))



class FocalLoss(nn.Module):
    def __init__(self, gamma = 0., weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight # means alpha
        self.epsilon = 1e-7
        
    def forward(self, x, y):

        one_hot_label = F.one_hot(y, x.shape[1])

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