import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.functional as F


class ExamplarLoss(nn.Module):

    def __init__(self,batch_size,device):
        super(ExamplarLoss,self).__init__()
        self.batch_size=batch_size
        self.loss_mask=torch.zeros([self.batch_size*2,self.batch_size*2],device=device)
        for i in range(self.batch_size):
            self.loss_mask[2*i,2*i+1]=1
            self.loss_mask[2 * i+1, 2 * i] = 1
        N = 2 * self.batch_size
        self.mask=1-torch.eye(N,N,device=device)



    def forward(self,latent1,latent2,eplision=1):
        '''
        :param latent1:
        :param latent2:
        :return:
        '''
        latent=torch.stack([latent1,latent2],dim=1)

        N = 2*self.batch_size
        latent = latent.reshape((N,-1))
        latent_i=torch.unsqueeze(latent,dim=-1).contiguous() #2N*L*1


        latent_i=latent_i.repeat((1,1,N)).contiguous() #2N*L*2N
        latent_j=torch.transpose(latent_i,0,2).contiguous()#2N*L*2N

        S=torch.sum(latent_i*latent_j,dim=1)    #2N*2N
        norm_i=torch.norm(latent_i,p=2,dim=1)
        norm_j=torch.norm(latent_j,p=2,dim=1)
        S=S/(eplision*norm_i*norm_j)            #2N*2N


        S=S*self.mask

        m=nn.LogSoftmax(dim=-1)
        loss=m(S)
        loss=torch.sum(loss*self.loss_mask)/N
        return -loss