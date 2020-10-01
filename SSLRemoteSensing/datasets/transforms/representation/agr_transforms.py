'''
@anthor: Wenyuan Li
@desc: Transforms for self-supervised
@date: 2020/5/22
'''
from .builder import build_transforms
import torchvision
import numpy as np

class AGRTransforms(object):

    def __init__(self,transforms_cfg:dict,shortcut_cfg,**kwargs):

        self.transforms=[]
        for param in transforms_cfg.values():
            self.transforms.append(build_transforms(**param))

        shortcut_transforms=[]
        for param in shortcut_cfg.values():
            shortcut_transforms.append(build_transforms(**param))
        shortcut_transforms=torchvision.transforms.RandomOrder(shortcut_transforms)
        self.shortcut_transforms=torchvision.transforms.Compose([shortcut_transforms])

    def forward(self,img):
        agr_label = np.random.randint(0, len(self.transforms), 1)[0]
        post_img=self.transforms[agr_label](img)
        post_img=self.shortcut_transforms(post_img)
        return post_img,agr_label