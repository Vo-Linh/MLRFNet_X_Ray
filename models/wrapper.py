import os
from argparse import Namespace

import numpy as np
import torch
from torchmetrics.classification import MultilabelAUROC
from tqdm import tqdm
import pandas as pd

from .helper_setup import load_weights


class NetworkWrapper:
    """A wrapper class for training and evaluating a PyTorch model.

    Args:
        net (nn.Module): The PyTorch model to be trained or evaluated.
        losses (list[nn.Module]): The loss function to be used.
        iter_per_epoch (int): The number of iterations per epoch.
        opt (argparse.Namespace): The parsed command-line arguments.
        config (dict): The configuration dictionary.

    Attributes:
        net (nn.Module): The PyTorch model.
        iter_per_epoch (int): The number of iterations per epoch.
        gpu_ids (list[int]): The IDs of the GPUs to use for training.
        device (torch.device): The device to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        best_acc (float): The best accuracy achieved so far.
        config (dict): The configuration dictionary.
        losses (nn.Module): The loss function.
        num_classes (int): The number of classes in the dataset.
        metric_AUC (MultilabelAUROC): The AUROC metric.
    """

    def __init__(self, net, losses, iter_per_epoch, opt, config, list_classes):
        self.net = net
        self.iter_per_epoch = iter_per_epoch
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        optim_dict, lr_dict, start_epoch = self.load_ckpt(opt)
        opt.start_epoch = start_epoch
        self.set_optimizer(opt, config, optim_dict, lr_dict)
        self.best_acc = 0.0
        self.config = config
        self.losses = losses
        self.list_classes = list_classes
        self.num_classes = self.config['HEAD']['NUM_CLASSES']
        self.eval_mode = 'micro'
        self.metric_AUC = MultilabelAUROC(average=self.eval_mode,
                                          num_labels=self.num_classes).to(self.device)

    def set_optimizer(self, opt, config, optim_dict, lr_dict):
        if not opt.is_Train:
            return

        # Initialize optimizer
        lr = config['TRAINING']['LR']
        weight_decay = config['TRAINING']['WEIGHT_DECAY']
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay)
        print('Setup Adam optimizer(lr={},wd={})'.format(lr, weight_decay))

        # Reload optimizer state dict if exists
        if optim_dict:
            self.optimizer.load_state_dict(optim_dict)

        # Initialize lrd scheduler
        def scheduler_lambda(epoch): return 0.9 ** epoch
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=scheduler_lambda)

    def recursive_todevice(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float)
        elif isinstance(x, dict):
            return {k: self.recursive_todevice(v) for k, v in x.items()}
        else:
            return [self.recursive_todevice(c) for c in x]

    def optim_step_(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_epoch(self, epoch, data_loader, log_print, log_metric_mlflow):
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
            data_loader (torch.utils.data.DataLoader): The training data loader.
            log_print (function): A function to print the training log.
            log_metric_mlflow (function): A function to logging  the training to MLflow.

        Returns:
            Namespace: A namespace containing the training loss and accuracy.
        """
        self.net.train()
        self.metric_AUC.reset()
        epoch_loss = {
            "loss_cls": [],
            "loss_vib": []
        }

        for data in tqdm(data_loader, ascii=' >='):
            image, mask = self.recursive_todevice(data)

            out_net = self.net(image)

            if len(out_net) == 3:
                pred, mu, std = out_net

                loss_cls = self.losses["loss_cls"](pred, mask)
                loss_vib = self.losses["loss_vib"](mu, std)
                epoch_loss["loss_cls"].append(loss_cls.item())
                epoch_loss['loss_vib'].append(loss_vib.item())
                loss = loss_cls + loss_vib
            else:
                pred = out_net
                loss_cls = self.losses["loss_cls"](pred, mask)
                loss = loss_cls
            self.optim_step_(loss)

            epoch_loss["loss_cls"].append(loss_cls.item())

            self.metric_AUC.update(pred, mask.long())

        if self.lr_scheduler:
            self.lr_scheduler.step()
            for param_group in self.optimizer.param_groups:
                print(f"LR: {param_group['lr']}")
                
        if len(out_net) == 3:
            kl_loss = np.mean(epoch_loss['loss_vib'])
            log_print(f'KL loss TRAIN {kl_loss:.4f}')
            log_metric_mlflow('KL loss TRAIN', kl_loss)

        acc_train = self.metric_AUC.compute()
        if self.eval_mode == 'none':
            acc_train = torch.mean(acc_train)

        log_print(
            f'TRAIN loss_cls={np.mean(epoch_loss["loss_cls"]):.4f} Acc={acc_train:.4f}')
        metrics = Namespace(train_loss_cls=np.mean(epoch_loss["loss_cls"]),
                            train_acc=acc_train)

        return metrics

    def eval_model(self, epoch, data_loader, log_print, log_metric_mlflow):
        """Evaluate the model on the validation set.

        Args:
            epoch (int): The current epoch.
            data_loader (torch.utils.data.DataLoader): The validation data loader.
            log_print (function): A function to print the evaluation log.
            log_metric_mlflow (function): A function to logging  the evaluation to MLflow.
        Returns:
            Namespace: A namespace containing the validation loss and accuracy.
            METRICS: Loss and AUROC
        """
        self.net.eval()
        self.metric_AUC.reset()
        epoch_val_loss = {
            "loss_cls": [],
            "loss_vib": []
        }

        with torch.no_grad():
            for data in tqdm(data_loader, ascii=' >='):
                (image), mask = self.recursive_todevice(data)
                out_net = self.net.forward(image)
                if len(out_net) == 3:
                    pred, mu, std = out_net

                    loss_cls = self.losses["loss_cls"](pred, mask)
                    loss_vib = self.losses["loss_vib"](mu, std)
                    epoch_val_loss["loss_cls"].append(loss_cls.item())
                    epoch_val_loss['loss_vib'].append(loss_vib.item())

                else:
                    pred = out_net
                    val_loss_cls = self.losses["loss_cls"](pred, mask)
                    epoch_val_loss["loss_cls"].append(val_loss_cls.item())

                self.metric_AUC.update(pred, mask.long())

        acc_val = self.metric_AUC.compute()
        if self.eval_mode == 'none':
            dict_acc_val = dict(zip(self.list_classes, acc_val.tolist()))
            df = pd.DataFrame(dict_acc_val.items(), columns=['Finding', 'Probability'])
            df = pd.DataFrame(dict_acc_val.items(), columns=['Finding', 'Probability'])
            log_print(f'AUC per Class: {df.to_string()}')
            acc_val = torch.mean(acc_val)

        if len(out_net) == 3:
            kl_loss = np.mean(epoch_val_loss['loss_vib'])
            log_print(f'KL loss VALID {kl_loss:.4f}')
            log_metric_mlflow('KL loss VALID', kl_loss)

        log_print(
            f'VALID loss_cls={np.mean(epoch_val_loss["loss_cls"]):.4f} Acc={acc_val:.4f}')
        metrics = Namespace(val_loss_cls=np.mean(epoch_val_loss["loss_cls"]),
                            val_acc=acc_val)
        return metrics

    def save_ckpt(self, epoch, out_dir, last_ckpt=False, best_acc=None, is_best=False):
        ckpt = {'last_epoch': epoch, 'best_acc': best_acc, 'model_dict': self.net.state_dict(),
                'optimizer_dict': self.optimizer.state_dict(),
                'lr_scheduler_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None}

        if last_ckpt:
            ckpt_name = 'last_ckpt.pth'
        elif is_best:
            ckpt_name = 'best_ckpt.pth'
        else:
            ckpt_name = 'ckpt_ep{}.pth'.format(epoch + 1)
        ckpt_path = os.path.join(out_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, config):
        ckpt_path = config.checkpoint
        if ckpt_path is None:
            return None, None, 0

        ckpt = load_weights(ckpt_path, self.device)
        start_epoch = ckpt['last_epoch'] + 1
        self.best_acc = ckpt['best_acc']
        print(
            'Load ckpt from {}, reset start epoch {}, best acc {}'.format(ckpt_path, config.start_epoch, self.best_acc))

        # Load net state
        model_dict = ckpt['model_dict']
        if len(model_dict.items()) == len(self.net.state_dict()):
            print('Reload all net parameters from weights dict')
            self.net.load_state_dict(model_dict)
        else:
            print('Reload part of net parameters from weights dict')
            self.net.load_state_dict(model_dict, strict=False)

        # Load optimizer state
        return ckpt['optimizer_dict'], ckpt['lr_scheduler_dict'], start_epoch

    def print_net_(self):
        for k, v in self.net.state_dict().items():
            print(k, v.size())

    def init_net(self, gpu_ids=[]):
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            print("Let's use", len(gpu_ids), "GPUs!")
            self.net.to(gpu_ids[0])
            # multi-GPUs
            self.net = torch.nn.DataParallel(self.net, gpu_ids)

        return self.net
