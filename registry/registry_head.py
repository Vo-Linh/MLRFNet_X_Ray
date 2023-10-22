from models.heads import mrfc

def define_head(config):

    if config['MRFC']['TYPE'] == 'MRFC':
        head = mrfc.MRFC(in_channels=config['HEAD']['IN_CHANNELS'],
                         num_classes=config['HEAD']['NUM_CLASSES'],
                         lam=config['HEAD']['LAM'])
        
    return head