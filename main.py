
import os
from typing import Any, Text, Mapping

import torch
from torchvision.transforms import transforms
import mlflow
from mlflow import log_metric, log_metrics,log_param, log_params, log_artifacts, set_experiment

from helper import parser
from models.losses.kl_loss import KLLoss
from models.wrapper import NetworkWrapper
from models.helper_setup import load_weights

from registry.registry_losses import define_loss
from registry.registry_backbone import define_backbone
from registry.registry_neck import define_neck
from registry.registry_head import define_head
from registry.resgistry_classifier import define_classifier
from registry.resgistry_selection import define_selection
from registry.registry_dataset import define_dataset

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
    def log_metric_mlflow(name, value, multi = False):
        if  multi:
            metrics = dict(zip(name, value.tolist()))
            return log_metrics(metrics)
        return log_metric(name, value)

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
            transforms.Resize((224,224)),
            # transforms.RandomResizedCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    data_config_train = config['DATA_TRAIN']
    train_data = define_dataset(data_config_train, data_transforms['train'])
    
    data_config_val = config['DATA_VAL']
    val_data = define_dataset(data_config_val, data_transforms['val'])

    if config['DATA_TEST']['USE'] == 'True':
        data_config_test = config['DATA_TEST']
        test_data = define_dataset(data_config_test, data_transforms['val'])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['EVALUATION']['BATCH_SIZE'],
                                             shuffle=False, num_workers=data_config_test['NUM_WORKERS'],
                                             pin_memory=False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['TRAINING']['BATCH_SIZE'],
                                               shuffle=True, num_workers=data_config_train['NUM_WORKERS'],
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['EVALUATION']['BATCH_SIZE'],
                                             shuffle=False, num_workers=data_config_val['NUM_WORKERS'],
                                             pin_memory=True)
    

    backbone = define_backbone(config=config)
    neck = define_neck(config=config)
    head = define_head(config=config)
    selection = define_selection(config=config)

    model = define_classifier(config, backbone, neck, head)

    loss_cls = define_loss(config=config)
    loss_vib = KLLoss(loss_weight=0.01)
    losses = {
        "loss_cls": loss_cls,
        "loss_vib": loss_vib
    }

    iter_per_epoch = len(train_loader)
    list_classes = config['DATA_TRAIN']['CLASS_COL']
    wrapper = NetworkWrapper(model, losses, iter_per_epoch, opts, config, list_classes)
    model = wrapper.init_net(opts.gpu_ids)
    log_print(
        'Load datasets from {}: train_set={} val_set={}'.format(data_config_train['IMG_DIR'],
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
        train_metrics = wrapper.train_epoch(
            epoch, train_loader, log_print, log_metric_mlflow)
        wrapper.save_ckpt(epoch, os.path.dirname(log_file),
                          best_acc=best_acc, last_ckpt=True)

        log_metric('Train Loss Cls', train_metrics.train_loss_cls)
        log_metric('Train AUC', train_metrics.train_acc)

        # Save network periodically
        if (epoch + 1) % opts.save_step == 0:
            wrapper.save_ckpt(epoch, os.path.dirname(
                log_file), best_acc=best_acc)

        # Eval on validation set
        val_metrics = wrapper.eval_model(epoch, val_loader, log_print, log_metric_mlflow)
        log_print(
            f'\nEvaluate {epoch + 1} \nLoss cls :{val_metrics.val_loss_cls:.4f} Acc:{val_metrics.val_acc:.4f}')

        log_metric('Valid Loss Cls', val_metrics.val_loss_cls)
        log_metric('val AUC', val_metrics.val_acc)

        # Save the best model
        mean_acc = val_metrics.val_acc
        if mean_acc > best_acc:
            best_acc = mean_acc
            wrapper.save_ckpt(epoch, os.path.dirname(
                log_file), best_acc=best_acc, is_best=True)
            log_print('>>Save best model: epoch={} AUROC:{:.4f}'.format(
                epoch + 1, mean_acc))

    # Testing Phase
    if config['DATA_TEST']['USE'] == 'True':
        weights_dict = load_weights(f'results_directory/{opt.name}/best_ckpt.pth',
                                    opt.gpu_ids)
        model_dict = weights_dict['model_dict']

        if len(model_dict.items()) == len(model.state_dict()):
                best_epoch = weights_dict['last_epoch'] + 1
                print(f'Load best epoch {best_epoch}')
                print('Reload all net parameters from weights dict')
                model.load_state_dict(model_dict)
        else:
            print('Reload part of net parameters from weights dict')
            model.load_state_dict(model_dict, strict=False)

        test_metrics = wrapper.eval_model(epoch+1, test_loader, log_print, log_metric_mlflow)
        log_print(
            f'\nLoss cls :{test_metrics.val_loss_cls:.4f} Acc:{test_metrics.val_acc:.4f}')
    
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
