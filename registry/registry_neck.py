from models.necks import fpn_eca

def define_neck(config):

    if config['NECK']['TYPE'] == 'FPN_ECA':
        neck = fpn_eca.FPN_ECA(config['NECK']['IN_CHANNELS'])

    return neck