
import os
from typing import Any, Text, Mapping

import torch
from torchvision.transforms import transforms
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts, set_experiment

from helper import parser
from datasets.chest14 import Chest14Dataset
from models.classifiers.mlrfnet import MLRFNet
from models.wrapper import NetworkWrapper

from registry.registry_losses import define_loss
from registry.registry_backbone import define_backbone
from registry.registry_neck import define_neck
from registry.registry_head import define_head

# =====================================================
# TODO
# 1. Setup tracking metrics via MLFlow
# 2. Use torchmetrics evaluate model AUROC
# 3. Remove sigmoid activation in last layer of model
# 4. Set name of experience on MLFLow
# 5. Link module to registry
# Next step
# 1. Convert output of metrics from Avg to each of Class
# =====================================================

# setup log in mlflow
set_experiment("MLRFNet_X_Ray")
mlflow.autolog()


def main(opts: Any, config: Mapping[Text, Any]) -> None:
    """Runs the training.
    Args:
        opts (Any): Options specifying the training configuration.
    """
    log = open(log_file, 'a')
    def log_print(ms): return parse.log(ms, log)

    # Configure dataloader
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation([-5, 5]),
            transforms.ColorJitter(
                brightness=0.9, contrast=0.9, saturation=1.1, hue=0.05),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
             transforms.RandomResizedCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    data_config = config['DATA']
    train_data = Chest14Dataset(label_csv=data_config['LABEL_TRAIN'],
                                img_dir=data_config['IMG_DIR'],
                                transform=data_transforms['train'])
    val_data = Chest14Dataset(label_csv=data_config['LABEL_VAL'],
                              img_dir=data_config['IMG_DIR'],
                              transform=data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['TRAINING']['BATCH_SIZE'],
                                               shuffle=True, num_workers=data_config['NUM_WORKERS'],
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['EVALUATION']['BATCH_SIZE'],
                                             shuffle=False, num_workers=data_config['NUM_WORKERS'],
                                             pin_memory=True)

    backbone = define_backbone(config=config)
    neck = define_neck(config=config)
    head = define_head(config=config)

    model = MLRFNet(backbone, neck, head)

    loss = define_loss(config=config)

    iter_per_epoch = len(train_loader)
    wrapper = NetworkWrapper(model, loss, iter_per_epoch, opts, config)
    model = wrapper.init_net(opts.gpu_ids)
    log_print(
        'Load datasets from {}: train_set={} val_set={}'.format(data_config['IMG_DIR'],
                                                                len(train_data),
                                                                len(val_data)))

    best_acc = wrapper.best_acc
    n_epochs = config['TRAINING']['EPOCHS']
    log_print('Start training from epoch {} to {}, best acc: {}'.format(
        opts.start_epoch, n_epochs, best_acc))

    # Log param in MLflowtrain_epoch
    log_param("EPOCHS", n_epochs)
    log_params({"LR": config['TRAINING']['LR'],
                "LR_DECAY_FACTOR": config['TRAINING']['LR_DECAY_FACTOR'],
                "LR_DECAY_STEP": config['TRAINING']['LR_DECAY_STEP']})

    for epoch in range(opts.start_epoch, n_epochs):
        log_print(f'>>> Epoch {epoch + 1}')
        train_metrics = wrapper.train_epoch(epoch, train_loader, log_print)
        wrapper.save_ckpt(epoch, os.path.dirname(log_file),
                          best_acc=best_acc, last_ckpt=True)

        log_metric('Train Loss', train_metrics.train_loss)
        log_metric('Train Acc', train_metrics.train_acc)

        # Save network periodically
        if (epoch + 1) % opts.save_step == 0:
            wrapper.save_ckpt(epoch, os.path.dirname(
                log_file), best_acc=best_acc)

        # Eval on validation set
        val_metrics = wrapper.eval_model(epoch, val_loader, log_print)
        log_print(
            f'\nEvaluate {epoch + 1} \nLoss:{val_metrics.val_loss:.4f} Acc:{val_metrics.val_acc:.4f}')

        log_metric('Valid Loss', val_metrics.val_loss)
        log_metric('val Acc', val_metrics.val_acc)

        # Save the best model
        mean_acc = val_metrics.val_acc
        if mean_acc > best_acc:
            best_acc = mean_acc
            wrapper.save_ckpt(epoch, os.path.dirname(
                log_file), best_acc=best_acc, is_best=True)
            log_print('>>Save best model: epoch={} AUROC:{:.4f}'.format(
                epoch + 1, mean_acc))


if __name__ == '__main__':
    parse = parser.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)

    config = parser.read_yaml_config(opt.config)
    mlflow.set_tag("mlflow.runName", opt.name)

    main(opts=opt, config=config)

    # Log an artifact (output file)
    if not os.path.exists("mlflow"):
        os.makedirs("mlflow")
    log_artifacts("mlflow")
