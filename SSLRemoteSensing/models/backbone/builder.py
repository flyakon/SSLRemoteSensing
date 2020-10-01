from .resnet import get_resnet
from .vgg import get_vgg
import torch

def build_backbone(name='resnet50',**kwargs):
    if name.startswith('resnet'):
        model=get_resnet(name,**kwargs)
    elif name.startswith('vgg'):
        model=get_vgg(name,**kwargs)
    else:
        raise NotImplementedError(r'''{0} is not an available values. \
                                          Please choose one of the available values in
                                           [resnet18, reset50, resnet101, resnet152,vgg11,vgg16]'''.format(name))
    return model