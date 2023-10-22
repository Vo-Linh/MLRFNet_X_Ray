from models.backbones import res2net
import torch.utils.model_zoo as model_zoo


def define_backbone(config):

    if config['BACKBONE']['TYPE'] == 'Res2Net':
        backbone = res2net.Res2Net(res2net.Bottle2neck,
                                   layers=config['BACKBONE']['LAYERS'],
                                   baseWidth=config['BACKBONE']['BASEWIDTH'],
                                   scale=config['BACKBONE']['SCALE'])
        backbone.load_state_dict(model_zoo.load_url(
            config['BACKBONE']['PRETRAIN']))
        
    return backbone