'''
@anthor: Wenyuan Li
@desc: Transforms for self-supervised
@date: 2020/5/15
'''
import torch
import cv2
import torchvision
import torchvision.transforms.functional as F
import numpy as np
from skimage import util as sk_utils
import random
from PIL import Image
from SSLRemoteSensing.datasets.transforms.representation.transforms import ColorJitter,HorizontalFlip,VerticalFlip,Rotate
import cv2
class InpaintingTransforms(object):

    def __init__(self,min_cover_ratio=0.2,max_cover_ratio=1./3,
                 brightness=0.3,contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.3,0.3)):
        self.min_conver_ratio=min_cover_ratio
        self.max_cover_ratio=max_cover_ratio
        self.colorJitter=ColorJitter(brightness=brightness,
                                     contrast=contrast,saturation=saturation,hue=hue)

    def __call__(self,img):
        return self.forward(img)

    def random_block(self,block):
        '''
        对block进行随机的操作
        :param block: [3,height,width]
        :return:
        '''

        idx = np.random.randint(0, 4, (1,), dtype=np.int64)[0]
        if idx==0:

            block=np.flip(block,0)
        elif idx==1:
            block = np.flip(block, 1)
        else:
            k=np.random.randint(1, 4, (1,), dtype=int)[0]
            k=int(k)
            block=np.rot90(block,k)
        return block

    def forward(self,img):
        dtype=np.ndarray
        if isinstance(img,torch.Tensor):
            img=img.numpy()
            dtype=torch.Tensor
        elif isinstance(img,Image.Image):
            img=np.array(img)
            dtype=Image.Image
        ratio=random.random()
        ratio=self.min_conver_ratio+ratio*(self.max_cover_ratio-self.min_conver_ratio)
        img_height,img_width=img.shape[0:2]
        crop_height=int(img_height*ratio)
        crop_width=int(img_width*ratio)
        crop_size=min(crop_height,crop_width)
        x=np.random.randint(0,img_width-crop_size,1)[0]
        y=np.random.randint(0,img_height-crop_size,1)[0]

        img[y:y+crop_size,x:x+crop_size]=\
            self.random_block(self.colorJitter(np.copy(img[y:y+crop_size,x:x+crop_size])))
        if dtype==torch.Tensor:
            img=torch.from_numpy(img)
        elif dtype==Image.Image:
            img=Image.fromarray(img)
        return img


if __name__=='__main__':
    import cv2

    img_files=r'F:\uvp_data\total\dior_data\00256_288_0.jpg'
    result_file=r'G:\other\inpainting.png'
    img=Image.open(img_files)
    img=F.resize(img,(256,256))
    trans=InpaintingTransforms(min_cover_ratio=0.6,max_cover_ratio=0.7)
    result=trans.forward(img)
    result=np.array(result)
    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.imwrite(result_file,result)