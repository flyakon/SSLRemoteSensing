import torch
import os
import shutil
import cv2
from skimage import measure
import numpy as np
from skimage import io
import argparse


isprs_map={0:(255,255,255),
           1:(0,0,255),
           2:(0,255,255),
           3:(0,255,0),
           4:(255,255,0),
           5:(255,0,0)}


def voc_colormap(N=21):
    def bitget(val, idx):
        return ((val & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        cmap[i, :] = [r, g, b]
    return cmap

VOC_COLOR_MAP=voc_colormap(21)

def load_model(model_path,current_epoch=None,prefix='cub_model'):

    '''
    载入模型,默认model文件夹中有一个latest.pth文件
    :param state_dict:
    :param model_path:
    :return:
    '''
    if os.path.isfile(model_path):
        model_file=model_path
    else:
        if current_epoch is None:
            model_file=os.path.join(model_path,'latest.pth')
        else:
            model_file = os.path.join(model_path, '%s_%d.pth'%(prefix,current_epoch))
    if not os.path.exists(model_file):
        print('warning:%s does not exist!'%model_file)
        return None,0,0
    print('start to resume from %s' % model_file)

    state_dict=torch.load(model_file)

    try:
        glob_step=state_dict.pop('gobal_step')
    except KeyError:
        print('warning:glob_step not in state_dict.')
        glob_step=0
    try:
        epoch=state_dict.pop('epoch')
    except KeyError:
        print('glob_step not in state_dict.')
        epoch=0

    return state_dict,epoch+1,glob_step

def save_model(model,model_path,epoch,global_step,prefix='cub_model',max_keep=10):

    if isinstance(model,torch.nn.Module):
        state_dict=model.state_dict()
    else:
        state_dict=model
    state_dict['epoch']=epoch
    state_dict['gobal_step']=global_step

    model_file=os.path.join(model_path,'%s_%d.pth'%(prefix,epoch))
    torch.save(state_dict,model_file)
    shutil.copy(model_file,os.path.join(model_path,'latest.pth'))

    # if epoch>max_keep:
    # 	for i in range(0,epoch-max_keep):
    # 		model_file=os.path.join(model_path,'%s_%d.pth'%(prefix,epoch))
    # 		if os.path.exists(model_file):
    # 			os.remove(model_file)

def localization(img,thres=120):
    loc_list=[]
    condinate_set=set()
    label_mask=img>thres
    label_img = np.array(label_mask, np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    label_img = cv2.morphologyEx(label_img, cv2.MORPH_CLOSE, kernel)
    label_mask=label_img>0
    label = measure.label(label_mask)
    props = measure.regionprops(label)
    # img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    for prop in props:
        # if prop.area<=200:
        # 	continue
        bbox=prop.bbox
        if bbox in condinate_set:
            continue
        condinate_set.add(bbox)
        loc_list.append([bbox[1],bbox[0],bbox[3],bbox[2]])
    loc_list=np.array(loc_list)
    return loc_list

def calc_iou(prediction,gt):
    inter_xmin=np.maximum(prediction[:,0],gt[:,0])
    inter_ymin=np.maximum(prediction[:,1],gt[:,1])
    inter_xmax = np.minimum(prediction[:, 2], gt[:, 2])
    inter_ymax = np.minimum(prediction[:, 3], gt[:, 3])
    inter_height=np.maximum(inter_ymax-inter_ymin,0)
    inter_width=np.maximum(inter_xmax-inter_xmin,0)
    inter_area=inter_height*inter_width

    total_area=(prediction[:,3]-prediction[:,1])*(prediction[:,2]-prediction[:,0])+\
               (gt[:,3]-gt[:,1])*(gt[:,2]-gt[:,0])-inter_area
    iou=inter_area/total_area
    return iou

def rectangle(img,bndboxes,color):
    img_height, img_width, _ = img.shape
    img = cv2.rectangle(img, (int(bndboxes[0] * img_width), int(bndboxes[1] * img_height)),
                        (int(bndboxes[2] * img_width), int(bndboxes[3] * img_height)), color, 2)
    return img

def vis_info(img,gt_box,pred_box,result_path,file_name):

    img=img*255
    if isinstance(img,torch.Tensor):
        img=img.cpu().numpy()
    img=img.astype(np.uint8)
    img=np.transpose(img,[1,2,0])
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    if isinstance(gt_box, torch.Tensor):
        gt_box=gt_box.cpu().numpy()
    img=rectangle(img,gt_box,[0,255,0])
    if isinstance(pred_box,torch.Tensor):
        pred_box = pred_box.cpu().detach().numpy()
    img = rectangle(img, pred_box, [0, 0, 255])

    cv2.imwrite(os.path.join(result_path,'{0}.jpg'.format(file_name)),img)

def vis_fcn_result(img:torch.Tensor,label:torch.Tensor,result:torch.Tensor,
                   result_path,file_name,as_binary=False):
    img=img.cpu().numpy()
    result=result.cpu().detach().numpy()
    img=img*255
    img=img.astype(np.uint8)
    img=np.transpose(img,(1,2,0))

    # result=result[1]>0.5
    if as_binary:
        result=np.argmax(result,axis=0)
        result = result * 127
        # result=np.where(result[1]>0.5,255,0)
    else:
        result=result[1]*255
    result=result.astype(np.uint8)


    label=label.cpu().numpy()*127
    label=label.astype(np.uint8)

    io.imsave(os.path.join(result_path, '{0}_label.jpg'.format(file_name)), label)
    io.imsave(os.path.join(result_path,'{0}_img.jpg'.format(file_name)),img)
    io.imsave(os.path.join(result_path, '{0}_result.png'.format(file_name)), result)

def vis_nap(img,block,idx_logits,result_path,global_step):
    img = img.cpu().detach().numpy()
    img=img[0]
    img = (img+1)/2 * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))

    block = block.cpu().detach().numpy()
    block=block[0]
    block = (block+1)/2* 255
    block = block.astype(np.uint8)
    block = np.transpose(block, (1, 2, 0))
    block_size,_,_=block.shape

    idx=idx_logits.cpu().detach().numpy()
    idx=np.argmax(idx,axis=-1)[0]

    row = np.mod(idx, 3) * block_size
    col = (idx // 3) * block_size
    recovery_img=np.copy(img)
    recovery_img[col:col+block_size,row:row+block_size,:]=block
    io.imsave(os.path.join(result_path, '{0}_img.jpg'.format(global_step)), img)
    io.imsave(os.path.join(result_path, '{0}_recovery.jpg'.format(global_step)), recovery_img)

def vis_nap_argu(img,block,result_path,global_step):
    img = img.cpu().detach().numpy()
    img=img[0]
    img = (img+1)/2 * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))

    block = block.cpu().detach().numpy()
    block=block[0]
    block = (block+1)/2* 255
    block = block.astype(np.uint8)
    block = np.transpose(block, (1, 2, 0))

    recovery_img=block
    io.imsave(os.path.join(result_path, '{0}_img.jpg'.format(global_step)), img)
    io.imsave(os.path.join(result_path, '{0}_recovery.jpg'.format(global_step)), recovery_img)

def vis_isprs_result(img,label,result,result_path,file_name):
    img = img.cpu().numpy()
    result = result.cpu().detach().numpy()
    img = img * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))

    img_height,img_width,_=img.shape
    result = np.argmax(result, axis=0)
    result_map=-np.ones([img_height,img_width,3])
    for i in range(6):
        result_map=np.where(result[:,:,np.newaxis]==i,isprs_map[i],result_map)
    assert (result_map==-1).any() == False
    result_map = result_map.astype(np.uint8)

    label = label.cpu().numpy()
    label_map=-np.ones([img_height,img_width,3])
    for i in range(6):
        label_map = np.where(label[:, :, np.newaxis] == i, isprs_map[i], label_map)
    label_map = label_map.astype(np.uint8)
    assert (label_map == -1).any() == False
    io.imsave(os.path.join(result_path, '{0}_label.jpg'.format(file_name)), label_map)
    io.imsave(os.path.join(result_path, '{0}_img.jpg'.format(file_name)), img)
    io.imsave(os.path.join(result_path, '{0}_result.jpg'.format(file_name)), result_map)

def vis_voc_result(img,label,result,result_path,file_name):
    img = img.cpu().numpy()
    result = result.cpu().detach().numpy()
    img = img * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))

    img_height,img_width,_=img.shape
    result = np.argmax(result, axis=0)
    result_map=-np.ones([img_height,img_width,3])
    for i in range(21):
        result_map=np.where(result[:,:,np.newaxis]==i,VOC_COLOR_MAP[i],result_map)
    assert (result_map==-1).any() == False
    result_map = result_map.astype(np.uint8)

    label = label.cpu().numpy()
    label_map=-np.ones([img_height,img_width,3])
    for i in range(21):
        label_map = np.where(label[:, :, np.newaxis] == i, VOC_COLOR_MAP[i], label_map)
    label_map = label_map.astype(np.uint8)
    assert (label_map == -1).any() == False
    io.imsave(os.path.join(result_path, '{0}_label.jpg'.format(file_name)), label_map)
    io.imsave(os.path.join(result_path, '{0}_img.jpg'.format(file_name)), img)
    io.imsave(os.path.join(result_path, '{0}_result.jpg'.format(file_name)), result_map)


def vis_agp_img(img,agu_img,result_path,global_step):
    img = img.cpu().detach().numpy()
    img=img[0]
    img = (img+1)/2 * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))

    agu_img = agu_img.cpu().detach().numpy()
    agu_img=agu_img[0]
    agu_img = (agu_img+1)/2* 255
    agu_img = agu_img.astype(np.uint8)
    agu_img = np.transpose(agu_img, (1, 2, 0))

    io.imsave(os.path.join(result_path, '{0}_img.jpg'.format(global_step)), img)
    io.imsave(os.path.join(result_path, '{0}_agu.jpg'.format(global_step)), agu_img)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower in ('none','null','-1'):
        return None
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__=='__main__':
    map=voc_colormap(21)
    print(map)