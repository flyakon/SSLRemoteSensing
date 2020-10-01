from .vr_losses import InpaintingLoss
from .examplar_loss import ExamplarLoss



import torch
import torchvision
import torch.nn as nn

losses_dict={'CrossEntropyLoss':nn.CrossEntropyLoss,'InpaintingLoss':InpaintingLoss,
             'ExamplarLoss':ExamplarLoss,'L1Loss':nn.L1Loss}


def builder_loss(name='CrossEntropyLoss',**kwargs):

    if name in losses_dict.keys():
        return losses_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))