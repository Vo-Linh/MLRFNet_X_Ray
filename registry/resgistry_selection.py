from models.selection import feature_selection, vib

def define_selection(config):
    if config['SELECTION']['TYPE'] == 'FPN_FeatureSelection':
        selection = feature_selection.FPN_FeatureSelection(config['SELECTION']['IN_CHANNELS'])

        return selection
    
    if config['SELECTION']['TYPE'] == 'FPN_VIB':
        selection = vib.FPN_VIB(config['SELECTION']['IN_CHANNELS'])

        return selection
