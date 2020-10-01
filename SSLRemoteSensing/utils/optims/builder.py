import torch.optim as optim

optim_dict={'Adam':optim.Adam,
            'SGD':optim.SGD}
lr_schedule_dict={'stepLR':optim.lr_scheduler.StepLR}
def build_optim(name='Adam',**kwargs):
    if name in optim_dict.keys():
        return optim_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))


def build_lr_schedule(name='stepLR',**kwargs):
    if name in lr_schedule_dict.keys():
        return lr_schedule_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))