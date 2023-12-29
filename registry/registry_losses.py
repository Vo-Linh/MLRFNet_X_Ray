from models.losses import biased_focal_loss, sigmoid_focal_loss, feature_selection_loss, kl_loss
from mlflow import log_params


def define_loss(config):

    if config['LOSS']['TYPE'] == 'SigmoidFocalLoss':
        loss = sigmoid_focal_loss.SigmoidFocalLoss(alpha=config['LOSS']['ALPHA'],
                                                   gamma=config['LOSS']['GAMMA'],
                                                   reduction=config['LOSS']['REDUCTION'],
                                                   loss_weight=config['LOSS']['LOSS_WEIGHT'])

        log_params({"LOSS": config['LOSS']['TYPE'],
                    'ALPHA': config['LOSS']['ALPHA'],
                    'GAMMA': config['LOSS']['GAMMA'],
                    'REDUCTION': config['LOSS']['REDUCTION']
                    })
        
    elif config['LOSS']['TYPE'] == 'BiasedFocalLoss':
        loss = biased_focal_loss.BiasedFocalLoss(beta=config['LOSS']['BETA'],
                                                 alpha=config['LOSS']['ALPHA'],
                                                 s=config['LOSS']['S'],
                                                 reduction=config['LOSS']['REDUCTION'],
                                                 loss_weight=config['LOSS']['LOSS_WEIGHT'])
        log_params({"LOSS": config['LOSS']['TYPE'],
                    'ALPHA': config['LOSS']['ALPHA'],
                    'S': config['LOSS']['S'],
                    'BETA': config['LOSS']['BETA'],
                    'REDUCTION': config['LOSS']['REDUCTION']
                    })
    elif config['LOSS']['TYPE'] == 'FeatureSelectionLoss':
        loss = feature_selection_loss.FeatureSelectionLoss(gamma=config['LOSS']['GAMMA'],
                                                           loss_weight=config['LOSS']['LOSS_WEIGHT'])
    
    elif config['LOSS']['TYPE'] == 'KLLoss':
        loss = kl_loss.KLLoss(reduction_override=config['LOSS']['REDUCT'],
                              loss_weight=config['LOSS']['LOSS_WEIGHT'])
        log_params({"LOSS": config['LOSS']['TYPE'],
                    'LOSS_WEIGHT': config['LOSS']['LOSS_WEIGHT'],
                    'REDUCTION': config['LOSS']['REDUCTION']
                    })
    return loss