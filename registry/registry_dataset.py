from datasets import chest14, chexpert

def define_dataset(config, transform):

    if config['TYPE'] == 'CheXpertDataset':
        dataset = chexpert.CheXpertDataset(csv_path=config['LABEL'],
                                           image_root_path=config['IMG_DIR'],
                                           ignore_index=config['IGN_LABEL'],
                                           train_cols=config['CLASS_COL'],
                                           transform=transform)
        return dataset
    
    if config['TYPE'] == 'Chest14Dataset':
        dataset = chest14.Chest14Dataset(label_csv=config['LABEL'],
                                         img_dir=config['IMG_DIR'],
                                         train_cols=config['CLASS_COL'],
                                         transform=transform)
        
        return dataset
        