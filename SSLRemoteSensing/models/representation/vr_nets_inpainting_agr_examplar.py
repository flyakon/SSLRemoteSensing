'''
@anthor: Wenyuan Li
@desc: Networks for self-supervised
@date: 2020/5/20
'''
import torch
import torch.nn as nn
import torchvision
from ..backbone.builder import build_backbone
from SSLRemoteSensing.datasets.datasets.representation.vr_dataset_inpainting_agr import InpaintingAGRDataset
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import torch.utils.data as data_utils
import SSLRemoteSensing.utils.utils as utils
from SSLRemoteSensing.utils.optims.builder import build_optim,build_lr_schedule
from SSLRemoteSensing.losses.builder import builder_loss

class VRNetsWithInpaintingAGRExamplar(nn.Module):

    def __init__(self,backbone_cfg:dict,
                 inpainting_head_cfg:dict,
                 agr_head_cfg:dict,
                 examplar_head_cfg:dict,
                 train_cfg:dict,
                 **kwargs):
        super(VRNetsWithInpaintingAGRExamplar,self).__init__()
        self.backbone=build_backbone(**backbone_cfg)
        self.build_arch(inpainting_head_cfg)
        self.build_agr_arch(agr_head_cfg)
        self.build_examplar_arch(examplar_head_cfg)
        self.train_cfg=train_cfg


    def build_examplar_arch(self,examplar_cfg):
        in_channels = examplar_cfg['in_channels']
        out_channels = examplar_cfg['out_channels']
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.examplar_fc = nn.Sequential(*[
            nn.Linear(in_channels,  out_channels*2),
            nn.Sigmoid(),
            nn.Linear(2*out_channels, out_channels)
        ])

    def build_agr_arch(self,agr_cfg):
        in_channels = agr_cfg['in_channels']
        num_classes=agr_cfg['num_classes']
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.agr_conv=nn.Sequential(*[
            nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ])
        self.agr_fc=nn.Sequential(*[
            nn.Linear(in_channels,2*in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2*in_channels,num_classes)
        ])

    def build_arch(self,head_cfg):

        in_channels=head_cfg['in_channels']
        out_channels=head_cfg['out_channels']
        feat_channels=head_cfg['feat_channels']

        self.trans_conv1=nn.Sequential(*[
            nn.Conv2d(in_channels, feat_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[0]),
            nn.ReLU(inplace=True)])
        self.trans_conv2 = nn.Sequential(*[
            nn.Conv2d(feat_channels[0], feat_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[1]),
            nn.ReLU(inplace=True)])
        self.trans_conv3 = nn.Sequential(*[
            nn.Conv2d(feat_channels[1], feat_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[2]),
            nn.ReLU(inplace=True)])
        self.trans_conv4 = nn.Sequential(*[
            nn.Conv2d(feat_channels[2], feat_channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[3]),
            nn.ReLU(inplace=True)])
        self.trans_conv5 = nn.Sequential(*[
            nn.Conv2d(feat_channels[3], feat_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[4]),
            nn.ReLU(inplace=True)])

        self.pred_conv=nn.Conv2d(feat_channels[4], out_channels, kernel_size=3, stride=1, padding=1)

    def forward_agr(self,pre_img,post_img):
        pre_logits,_=self.backbone(pre_img)
        post_logits,_ = self.backbone(post_img)
        x=torch.cat((pre_logits,post_logits),dim=1)

        x=self.agr_conv(x)
        x=self.avg_pool(x)
        x=torch.flatten(x,1)
        logits=self.agr_fc(x)
        return logits,pre_logits,post_logits


    def forward_inpainting(self, x, **kwargs):
        x,endpoints=self.backbone(x)

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv1(x)
        if 'block5' in endpoints.keys():
            x=x+endpoints['block5']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv2(x)
        if 'block4' in endpoints.keys():
            x = x + endpoints['block4']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv3(x)
        if 'block3' in endpoints.keys():
            x = x + endpoints['block3']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv4(x)
        if 'block2' in endpoints.keys():
            x = x + endpoints['block2']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv5(x)
        if 'block1' in endpoints.keys():
            x = x + endpoints['block1']

        logits=self.pred_conv(x)

        return logits

    def forward_examplar(self,pre,post):
        pre=self.avg_pool(pre)
        pre=torch.flatten(pre,1)
        pre=self.examplar_fc(pre)

        post = self.avg_pool(post)
        post = torch.flatten(post, 1)
        post = self.examplar_fc(post)
        return pre,post

    def forward(self,inpainting_img,pre_img,post_img):
        inpainting_logits=self.forward_inpainting(inpainting_img)
        agr_logits,pre_logits,post_logits=self.forward_agr(pre_img,post_img)
        pre_logits,post_logits=self.forward_examplar(pre_logits,post_logits)
        return inpainting_logits,agr_logits,pre_logits,post_logits


    def run_train_interface(self,**kwargs):
        batch_size=self.train_cfg['batch_size']
        device=self.train_cfg['device']
        num_epoch=self.train_cfg['num_epoch']
        num_workers=self.train_cfg['num_workers']
        checkpoint_path=self.train_cfg['checkpoints']['checkpoints_path']
        save_step=self.train_cfg['checkpoints']['save_step']
        log_path=self.train_cfg['log']['log_path']
        log_step=self.train_cfg['log']['log_step']
        with_vis=self.train_cfg['log']['with_vis']
        vis_path=self.train_cfg['log']['vis_path']
        self.to(device)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        train_dataset=InpaintingAGRDataset(**self.train_cfg['train_data'])
        train_dataloader=data_utils.DataLoader(train_dataset,batch_size,shuffle=True,num_workers=num_workers,
                                               drop_last=True)
        inpainting_criterion=builder_loss(**self.train_cfg['losses']['InpaintingLoss'])
        agr_criterion=builder_loss(**self.train_cfg['losses']['AGRLoss'])
        examplar_criterion=builder_loss(**self.train_cfg['losses']['ExamplarLoss'],
                                        device=device,batch_size=batch_size)
        loss_factors=self.train_cfg['losses']['factors']

        optimizer = build_optim(params=self.parameters(), **self.train_cfg['optimizer'])

        if 'lr_schedule' in self.train_cfg.keys():
            lr_schedule=build_lr_schedule(optimizer=optimizer,**self.train_cfg['lr_schedule'])

        state_dict, current_epoch, global_step = utils.load_model(checkpoint_path)
        if state_dict is not None:
            print('resume from epoch %d global_step %d' % (current_epoch, global_step))
            self.load_state_dict(state_dict, strict=True)

        summary = SummaryWriter(log_path)
        start_time = datetime.datetime.now()

        for epoch in range(current_epoch,num_epoch):
            for i, data in enumerate(train_dataloader):
                # data.to(device)
                global_step += 1
                input,pre_img,post_img,inpainting_label,attention_mask,agr_label=data
                input=input.to(device)
                pre_img=pre_img.to(device)
                post_img=post_img.to(device)
                inpainting_label=inpainting_label.to(device)
                attention_mask=attention_mask.to(device)
                agr_label=agr_label.to(device)

                self.train()
                outputs,agr_logits,pre_logits,post_logits = self.forward(input,pre_img,post_img)
                inpainting_loss = inpainting_criterion(outputs,inpainting_label,attention_mask)
                inpainting_loss=inpainting_loss*loss_factors[0]
                agr_loss=agr_criterion(agr_logits,agr_label)*loss_factors[1]
                examplar_loss=examplar_criterion(pre_logits,post_logits)*loss_factors[2]
                loss=inpainting_loss+agr_loss+examplar_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % log_step == 0:
                    end_time = datetime.datetime.now()
                    total_time = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
                    total_time = total_time / log_step / batch_size
                    fps = 1 / total_time * 1000
                    start_time = datetime.datetime.now()
                    # if args.is_vis:
                    #     utils.vis_nap(noise_img, block_recovery, block_logits, 'train_result', global_step)
                    print("[Epoch %d/%d] [Batch %d/%d] [inpainting loss:%f,agr loss:%f,examplar loss:%f,loss: %f] [fps:%f]" % (
                        epoch, num_epoch, i, len(train_dataloader),
                         inpainting_loss.item(),agr_loss.item(),examplar_loss.item(),loss.item(), fps))
                    summary.add_scalar('inpainting loss', inpainting_loss, global_step)
                    summary.add_scalar('agr loss', agr_loss, global_step)
                    summary.add_scalar('examplar loss', examplar_loss, global_step)
                    summary.add_scalar('total loss', loss, global_step)

            if 'lr_schedule' in self.train_cfg.keys():
                lr_schedule.step(epoch=epoch)
                summary.add_scalar('learning_rate',optimizer.state_dict()['param_groups'][0]['lr'],global_step)
            if epoch % save_step == 0:
                print('save model')
                utils.save_model(self, checkpoint_path,
                                 epoch, global_step, max_keep=200)






