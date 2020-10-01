import torch
import torch.utils.data as data_utils
import os
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms
import random
import torchvision.transforms.functional as F
import cv2
import sys
import collections
import numbers
import datetime

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class VerticalFlip():
    """Vertically flip the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """

    def __init__(self, p=0.5,with_idx=False):
        self.p = p
        self.with_idx=with_idx

    def __call__(self, img):
        """
        Args:
            img ( Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        label=0
        if random.random()<self.p:
            if isinstance(img, torch.Tensor):
                img = torch.flip(img, dims=(1,))
            elif isinstance(img, Image.Image):
                img = F.vflip(img)
            else:
                img = np.flipud(img)
            label=1
        if self.with_idx:
            return img,label
        else:
            return img

    def reverse(self, img,idx):
        '''
        :param img:
        :return:
        '''
        if idx==0:
            pass
        elif idx==1:
            img = torch.flip(img, dims=(0,))
        else:
            raise NotImplementedError('{0} not an available value in VerticalFlip!'.format(idx))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


# 图像垂直翻转
class HorizontalFlip():
    """Horizontally flip the given PIL Image randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """

    def __init__(self, p=0.5,with_idx=False):
        self.p = p
        self.with_idx=with_idx

    def __call__(self, img):
        """
        Args:
            img (ndarray Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        label=0
        if random.random()<self.p:
            if isinstance(img,torch.Tensor):
                img = torch.flip(img, dims=(1,))
            elif isinstance(img,Image.Image):
                img=F.hflip(img)
            else:
                img=np.fliplr(img)
            label=1
        if self.with_idx:
            return img,label
        else:
            return img

    def reverse(self, img,idx):
        '''
        把变化后的图像恢复
        :param img:
        :return:
        '''
        if idx == 0:
            pass
        elif idx==1:
            img = torch.flip(img, dims=(1,))
        else:
            raise NotImplementedError('{0} not an available value in HorizontalFlip!'.format(idx))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


# 图像旋转
class Rotate():
    """Rotate the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
            angle (int): Rotated angle [0,1,2,3]
        """

    def __init__(self, p=0.5,with_idx=False,angle=None):
        self.p = p
        self.with_idx=with_idx
        self.angle=angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.angle is None:
            angle=random.randint(0,4)*90
        else:
            angle=self.angle
        if isinstance(img, torch.Tensor):
            k=angle//90
            img = torch.rot90(img, k)
        elif isinstance(img, Image.Image):
            img = F.rotate(img,angle)
        else:
            k = angle // 90
            img=np.rot90(img,k)

        if self.with_idx:
            return img,angle
        else:
            return img

    def reverse(self, img,idx):
        img = torch.rot90(img, idx, (0, 1))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0,with_idx=True):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.with_idx=with_idx

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        if isinstance(img, torch.Tensor):
            img = Image.fromarray(img.numpy())
            img = transform(img)
            return torch.from_numpy(np.array(img))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            img = transform(img)
            return np.array(img)
        else:
            img = transform(img)
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

    def reverse(self, img):
        return img

class RandomCrop():
    """Horizontally flip the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """
    def __init__(self,crop_ratio_min=0.8,crop_ratio_max=0.95):
        self.crop_ratio_min=crop_ratio_min
        self.crop_ratio_max=crop_ratio_max

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self,img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if isinstance(img,np.ndarray):
            img=Image.fromarray(img)
        elif isinstance(img,torch.Tensor):
            img=Image.fromarray(img.numpy())
        img_height=img.height
        img_width=img.width
        self.crop_ratio=random.random()
        self.crop_ratio=self.crop_ratio*(self.crop_ratio_max-self.crop_ratio_min)+self.crop_ratio_min
        height=int(img_height*self.crop_ratio)
        width=int(img_width*self.crop_ratio)
        i, j, h, w = self.get_params(img,(height,width))
        img=F.crop(img, i, j, h, w)
        return img


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        if isinstance(img,np.ndarray):
            img=Image.fromarray(img)
        elif isinstance(img,torch.Tensor):
            img=Image.fromarray(img.numpy())

        img=F.resize(img, self.size, self.interpolation)
        return img


    def __repr__(self):

        return self.__class__.__name__ + '(size={0})'.format(self.size)

class Normal(object):

    def __call__(self,img):
        return img