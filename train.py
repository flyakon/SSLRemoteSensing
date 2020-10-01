import torch
import argparse
import numpy as np
import mmcv
from SSLRemoteSensing.models.builder import builder_models
from SSLRemoteSensing.utils import utils

parse=argparse.ArgumentParser()
# parse.add_argument('--config_file',
#             default=r'configs/vr_resnet50_inapinting_agr_cfg.py',type=str)
parse.add_argument('--config_file',default=r'configs/vr_vgg16_inapinting_agr_examplar_cfg.py',type=str)
#
parse.add_argument('--checkpoints_path',default=None,type=str)
parse.add_argument('--with_imagenet',default=None,type=utils.str2bool)
parse.add_argument('--log_path',default=None,type=str)

if __name__=='__main__':
    args = parse.parse_args()
    print(args)
    cfg = mmcv.Config.fromfile(args.config_file)
    if args.with_imagenet is not None:
        cfg['config']['backbone_cfg']['pretrained'] = args.with_imagenet
    models=builder_models(**cfg['config'])

    run_args={}

    models.run_train_interface(checkpoint_path=args.checkpoints_path,

                                  log_path=args.log_path)

