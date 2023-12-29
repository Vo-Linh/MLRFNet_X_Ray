from models.classifiers import mlrfnet, mlrfnet_fs, mlrfnet_vib

def define_classifier(config, backbone, neck, head, selection = None):

    if config['CLASSIFIER']['TYPE'] == 'MLRFNet':
        classifier = mlrfnet.MLRFNet(backbone, neck, head)

        return classifier

    if config['CLASSIFIER']['TYPE'] == 'MLRFNet_FS':
        classifier = mlrfnet_fs.MLRFNet_FS(backbone, neck, head, selection)
        return classifier
    
    if config['CLASSIFIER']['TYPE'] == 'MLRFNet_VIB':
        classifier = mlrfnet_vib.MLRFNet_VIB(backbone, neck, head, selection)

        return classifier