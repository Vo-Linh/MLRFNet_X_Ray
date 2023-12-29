from models.backbones import res2net
from models.backbones import resnet
import torch.utils.model_zoo as model_zoo


def define_backbone(config):

    if config['BACKBONE']['TYPE'] == 'Res2Net':
        backbone = res2net.Res2Net(res2net.Bottle2neck,
                                   layers=config['BACKBONE']['LAYERS'],
                                   baseWidth=config['BACKBONE']['BASEWIDTH'],
                                   scale=config['BACKBONE']['SCALE'])
        backbone.load_state_dict(model_zoo.load_url(
            config['BACKBONE']['PRETRAIN']),strict=False)
        
        return backbone
    
    if config['BACKBONE']['TYPE'] == 'ResNet':
        backbone= resnet.ResNet50()
        backbone.load_state_dict(model_zoo.load_url(
            config['BACKBONE']['PRETRAIN']),strict=False)
        
        return backbone

    