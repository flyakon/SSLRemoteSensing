import torch
import torch.nn as nn

class InpaintingLoss(object):

    def __init__(self,**kwargs):
        super(InpaintingLoss,self).__init__()
        self.criterion=nn.L1Loss(reduction='none')

    def __call__(self, logits,labels,attention_map):
        loss = self.criterion(logits, labels)
        loss = loss * attention_map
        count=torch.sum(attention_map)+1e-6
        loss = torch.sum(loss) / count
        return loss
