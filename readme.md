# Overview.
__Semantic Segmentation of Remote Sensing Images With Self-Supervised Multitask Representation Learning
[Paper](https://ieeexplore.ieee.org/abstract/document/9460820)__

Existing deep learning-based remote sensing images semantic segmentation methods require large-scale labeled datasets. However, the annotation of segmentation datasets is often too time-consuming and expensive. To ease the burden of data annotation, self-supervised representation learning methods have emerged recently. However, the semantic segmentation methods need to learn both high-level and low-level features, but most of the existing self-supervised representation learning methods usually focus on one level, which affects the performance of semantic segmentation for remote sensing images. In order to solve this problem, we propose a self-supervised multitask representation learning method to capture effective visual representations of remote sensing images. We design three different pretext tasks and a triplet Siamese network to learn the high-level and low-level image features at the same time. The network can be trained without any labeled data, and the trained model can be fine-tuned with the annotated segmentation dataset. We conduct experiments on Potsdam, Vaihingen dataset, and cloud/snow detection dataset Levir_CS to verify the effectiveness of our methods. Experimental results show that our proposed method can effectively reduce the demand of labeled datasets and improve the performance of remote sensing semantic segmentation. Compared with the recent state-of-the-art self-supervised representation learning methods and the mostly used initialization methods (such as random initialization and ImageNet pretraining), our proposed method has achieved the best results in most experiments, especially in the case of few training data. With only 10% to 50% labeled data, our method can achieve the comparable performance compared with random initialization.

![Overview](fig/overall_networks.png)

In this repository, we implement the training of self-supervised multi-task representation learning for remote sensing images with pytorch and generate pretrained models. With the code, you can also try on your own dataset by following the instructions below.



# Requriements

- python 3.6.7

- pytorch 1.7.0

- torchvision 0.6.0

- cuda 10.1

See also in [Requirements.txt](requirements.txt).

# Setup

1. Clone this repo.

   `git clone https://github.com/flyakon/SSLRemoteSensing.git`

   `cd SSLRemoteSensing`

2. Prepare the training data and put it into the specified folder, such as ".. / dataset / train_ data".

3. Modify the configs file [vr_vgg16_inapinting_agr_examplar_cfg.py](configs/vr_vgg16_inapinting_agr_examplar_cfg.py) to configure the training parameters.

   Some important training parameters:

   ```_
   backbone_cfg: which network ("vgg16_bn" or "resnet50 ") to choose as the backbone.
   inpainting_head_cfg, agr_head_cfg and examplar_head_cfg: network parameters corresponding to different pretext tasks.
   train_cfg: parameters corresponding to self-supervised representation learning.
   ```

4. Change the "--config_file" option to the location of   [vr_vgg16_inapinting_agr_examplar_cfg.py](configs/vr_vgg16_inapinting_agr_examplar_cfg.py) in [train.py](train.py) and run this file.

# Citation

If you find the code useful, please cite:

``````
@ARTICLE{9460820,
  author={Li, Wenyuan and Chen, Hao and Shi, Zhenwei},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Semantic Segmentation of Remote Sensing Images With Self-Supervised Multitask Representation Learning}, 
  year={2021},
  volume={14},
  number={},
  pages={6438-6450},
  doi={10.1109/JSTARS.2021.3090418}}
``````
