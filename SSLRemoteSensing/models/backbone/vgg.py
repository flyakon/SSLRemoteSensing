import torch
import torchvision
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from collections import OrderedDict

class NamedSequential(nn.Sequential):

    def forward(self, input: torch.Tensor):
        result_dict=OrderedDict()
        i=2
        for name,module in self._modules.items():
            # print(name,module)
            input = module(input)
            if type(module).__name__ =='MaxPool2d':
                key='block%d'%i
                result_dict[key]=input
                i+=1

        return input,result_dict




model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000,out_keys=None, init_weights=True):
        super(VGG, self).__init__()
        self.out_keys=out_keys
        self.features = features
        self.num_classes=num_classes
        if self.num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            print('initialize_weights')
            self._initialize_weights()

    def forward(self, x):
        x,endpoints = self.features(x)
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        if self.out_keys is None:
            endpoints = {}
        else:
            endpoints = {key: endpoints[key] for key in self.out_keys}
        return x,endpoints

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False,in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return NamedSequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

arch_cfg_dict={'vgg11':'A','vgg13':'B','vgg16':'D','vgg19':'E'}

def _vgg(arch, cfg, batch_norm, pretrained, progress,in_channels,num_classes, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm,in_channels=in_channels),num_classes=num_classes, **kwargs)

    if pretrained:
        if batch_norm:
            name='%s_bn'%arch
        else:
            name=arch
        state_dict = load_state_dict_from_url(model_urls[name],
                                              progress=progress)

        if in_channels != 3:
            keys = state_dict.keys()
            keys = [x for x in keys if 'features.0' in x]
            for key in keys:
                del state_dict[key]
        if num_classes is None:
            keys = state_dict.keys()
            keys = [x for x in keys if 'classifier' in x]
            for key in keys:
                del state_dict[key]
            model.load_state_dict(state_dict)
        elif num_classes != 1000:
            keys = state_dict.keys()
            keys = [x for x in keys if 'classifier.6' in x]
            for key in keys:
                del state_dict[key]
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    return model


def get_vgg(name='vgg16',pretrained=True,progress=True,
               num_classes=1000,out_keys=None,in_channels=3,**kwargs):
    '''
    Get resnet model with name.
    :param name: vgg model name
    :param pretrained: If True, returns a model pre-trained on ImageNet
    '''

    if pretrained and num_classes !=1000:
        print('warning: num_class is not equal to 1000, which will cause some parameters to fail to load!')
    if pretrained and in_channels !=3:
        print('warning: in_channels is not equal to 3, which will cause some parameters to fail to load!')
    batch_norm=True if 'bn' in name else False
    if batch_norm:
        name=name.replace('_bn','')
    print('batchnorm:{0}'.format(batch_norm))
    return _vgg(arch=name,cfg=arch_cfg_dict[name],batch_norm=batch_norm,pretrained=pretrained,progress=progress,
                num_classes=num_classes,out_keys=out_keys,in_channels=in_channels,**kwargs)

if __name__=='__main__':
    model=get_vgg('vgg16_bn',pretrained=True,num_classes=None,in_channels=3,
                     out_keys=('block2','block3','block4','block5','block6'))
    x=torch.rand([2,3,256,256])
    result,endponits=model.forward(x)
    print(result.shape)
    print(endponits['block6'].shape)
    print(endponits['block5'].shape)
    print(endponits['block4'].shape)
    print(endponits['block3'].shape)
    print(endponits['block2'].shape)

    print(endponits.keys())



