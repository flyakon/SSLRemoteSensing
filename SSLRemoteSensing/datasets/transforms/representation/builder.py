from .transforms import Rotate,HorizontalFlip,VerticalFlip,ColorJitter,\
    RandomCrop,Resize,Normal
import torchvision
from torchvision.transforms.transforms import RandomGrayscale
transforms_dict={
                 'RandomHorizontalFlip':HorizontalFlip,
                 'RandomVerticalFlip':VerticalFlip,
                 'Rotate':Rotate,
                 'ColorJitter':ColorJitter,
                 'RandomCrop':RandomCrop,
                 'ToTensor':torchvision.transforms.ToTensor,
                 'Resize':Resize,
                 'RandomGrayscale':torchvision.transforms.RandomGrayscale,
                 'Normal':Normal}


def build_transforms(name='NoiseTransforms',**kwargs):
    if name in transforms_dict.keys():
        return transforms_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in available values.'.format(name))