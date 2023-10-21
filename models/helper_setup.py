import torch

def load_weights(weights_dir, device, key='state_dict'):
    map_location = lambda storage, loc: storage.cuda(device.index) if torch.cuda.is_available() else storage
    weights_dict = None
    if weights_dir is not None:
        weights_dict = torch.load(weights_dir, map_location=map_location)
    return weights_dict