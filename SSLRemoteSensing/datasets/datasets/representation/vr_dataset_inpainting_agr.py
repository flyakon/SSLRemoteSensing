'''
@anthor: Wenyuan Li
@desc: Datasets for self-supervised
@date: 2020/5/15
'''
import torch
import torch.nn as nn
import torchvision
import glob
import os
import numpy as np
from skimage import io
from skimage import util as sk_utils
from PIL import Image
from SSLRemoteSensing.datasets.transforms.representation import inpainting_transforms,builder
import torch.utils.data as data_utils
from SSLRemoteSensing.datasets.transforms.representation.agr_transforms import AGRTransforms

class InpaintingAGRDataset(data_utils.Dataset):

    def __init__(self,data_path,data_format,inpainting_transforms_cfg,agr_transforms_cfg,
                 pre_transforms_cfg,post_transforms_cfg, img_size=256):
        super(InpaintingAGRDataset, self).__init__()

        self.data_files=glob.glob(os.path.join(data_path,data_format))

        self.img_size=img_size
        self.inpainting_transforms=inpainting_transforms.InpaintingTransforms(**inpainting_transforms_cfg)
        pre_transforms=[]
        for param in pre_transforms_cfg.values():
            pre_transforms.append(builder.build_transforms(**param))
        self.pre_transforms=torchvision.transforms.Compose(pre_transforms)

        post_transforms=[]
        for param in post_transforms_cfg.values():
            post_transforms.append(builder.build_transforms(**param))
        self.post_transforms=torchvision.transforms.Compose(post_transforms)

        self.agr_transforms =AGRTransforms(**agr_transforms_cfg)


    def __getitem__(self, item):

        img=Image.open(self.data_files[item])
        img=self.pre_transforms(img)

        inpainting_label=img

        pre_img=img
        post_img,agr_label=self.agr_transforms.forward(img)
        data=img
        data=self.inpainting_transforms(data)
        data=self.post_transforms(data)
        inpainting_label=self.post_transforms(inpainting_label)
        inpainting_mask=torch.abs(inpainting_label-data)
        pre_img=self.post_transforms(pre_img)
        post_img=self.post_transforms(post_img)
        agr_label=torch.tensor(agr_label,dtype=torch.int64)
        return data,pre_img,post_img, inpainting_label,inpainting_mask,agr_label


    def __len__(self):
        return len(self.data_files)



