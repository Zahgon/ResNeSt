##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import math
import atexit
import shutil
import functools
import threading
import numpy as np
import torch

from iopath.common.file_io import PathManager as PathManagerBase

__all__ = ['accuracy', 'AverageMeter', 'LR_Scheduler', 'mkdir',
           'torch_dist_sum', 'MixUpWrapper', 'save_checkpoint',
           'cached_log_stream', 'PathManager']

PathManager = PathManagerBase()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()


    def update(self, val, n=1):
        #self.val = val
        self.sum += val * n
        self.count += n






@master_only
def master_only_print(*args):
    """master-only print"""
    pass

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, quiet=False,
                 logger=None):
        self.mode = mode
        self.quiet = quiet
        self.logger = logger
        if not quiet:
            msg = 'Using {} LR scheduler with warm-up epochs of {}!'.format(self.mode, warmup_epochs)
            if self.logger:
                self.logger.info(msg)
            else:
                master_only_print()
        if mode == 'step':
            assert lr_step
        self.base_lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == 'cos':
            T = T - self.warmup_iters
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
        elif self.mode == 'poly':
            T = T - self.warmup_iters
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplementedError
        if epoch > self.epoch and (epoch == 0 or best_pred > 0.0):
            if not self.quiet:
                msg = '\n=>Epoch %i, learning rate = %.4f, \
                    previous best = %.4f' % (epoch, lr, best_pred)
                if self.logger:
                    self.logger.info(msg)
                else:
                    master_only_print()
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)



class MixUpWrapper(object):
    def __init__(self, alpha, num_classes, dataloader, device):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device


    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.mixup_loader(self.dataloader)

@master_only
def save_checkpoint(state, directory, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    pass

# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.

def mkdir(path):
    """Make directory at the specified local path with special error handling.
    """
    pass
