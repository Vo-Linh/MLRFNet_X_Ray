import os
from argparse import Namespace

import numpy as np
import torch
from torchmetrics.classification import MultilabelAUROC
from tqdm import tqdm

from .helper_setup import load_weights


class NetworkWrapper:
    """A wrapper class for training and evaluating a PyTorch model.

    Args:
        net (nn.Module): The PyTorch model to be trained or evaluated.
        losses (nn.Module): The loss function to be used.
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

    def __init__(self, net, losses, iter_per_epoch, opt, config):
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
        self.num_classes = self.config['HEAD']['NUM_CLASSES']
        self.metric_AUC = MultilabelAUROC(average='macro',
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
        self.lr_scheduler = None

    def recursive_todevice(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, dict):
            return {k: self.recursive_todevice(v) for k, v in x.items()}
        else:
            return [self.recursive_todevice(c) for c in x]

    def optim_step_(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_epoch(self, epoch, data_loader, log_print):
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
            data_loader (torch.utils.data.DataLoader): The training data loader.
            log_print (function): A function to print the training log.

        Returns:
            Namespace: A namespace containing the training loss and accuracy.
        """
        self.net.train()
        self.metric_AUC.reset()
        epoch_loss = []

        for data in tqdm(data_loader, ascii=' >='):
            image, mask = self.recursive_todevice(data)
            # mask = mask.long()
            pred = self.net(image)
            loss = self.losses(pred, mask)
            self.optim_step_(loss)

            epoch_loss.append(loss.item())
            self.metric_AUC.update(pred, mask.long())

        acc_train = self.metric_AUC.compute()
        log_print(
            f'TRAIN loss={np.mean(epoch_loss):.3f} Acc={acc_train:.3f}')
        metrics = Namespace(train_loss=np.mean(epoch_loss),
                            train_acc=acc_train)
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return metrics

    def eval_model(self, epoch, data_loader, log_print):
        """Evaluate the model on the validation set.

        Args:
            epoch (int): The current epoch.
            data_loader (torch.utils.data.DataLoader): The validation data loader.
            log_print (function): A function to print the evaluation log.

        Returns:
            Namespace: A namespace containing the validation loss and accuracy.
            METRICS: Loss and AUROC
        """
        self.net.eval()
        self.metric_AUC.reset()
        epoch_val_loss = []

        with torch.no_grad():
            for data in tqdm(data_loader, ascii=' >='):
                (image), mask = self.recursive_todevice(data)

                pred = self.net.forward(image)
                val_loss = self.losses(pred, mask)
                epoch_val_loss.append(val_loss.item())
                self.metric_AUC.update(pred, mask.long())

        acc_val = self.metric_AUC.compute()

        log_print(
            f'TRAIN loss={np.mean(epoch_val_loss):.3f} Acc={acc_val:.3f}')
        metrics = Namespace(val_loss=np.mean(epoch_val_loss),
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
            self.net = torch.nn.DataParallel(self.net, gpu_ids)  # multi-GPUs

        return self.net
