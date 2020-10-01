from .representation.vr_nets_inpainting_agr_examplar import VRNetsWithInpaintingAGRExamplar


models_dict={
             'VRNetsWithInpaintingAGRExamplar':VRNetsWithInpaintingAGRExamplar
}

def builder_models(name='VRNetsWithInpainting',**kwargs):
    if name in models_dict.keys():
        return models_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))