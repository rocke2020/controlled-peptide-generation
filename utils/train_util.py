import numpy as np
import torch
import random
import pickle, os
import pandas as pd
import torch.optim as optim
from .log_util import logger


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception as identifier:
        pass


def get_device(device_id=0):
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > device_id:
            device = f"cuda:{device_id}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    logger.info(f'device: {device}')
    return device


def get_weights_from_df(df, task_name='label', prefix='test'):
    num0 = len(df[df[task_name] == 0])
    num1 = len(df[df[task_name] == 1])
    max_num = max(num0, num1)
    logger.info(f'{prefix} dataset num0 {num0} vs num1 {num1}')
    weights = [max_num/num0, max_num/num1]
    logger.info(f'{prefix} dataset weights {weights}')
    return weights


def cal_log_interval(data_len, batch_size, total_log_times_num=2000):
    """ log 1k~10K times number is proper, maybe 2k, to timely view progress and also no waste too much time log! """
    proper_log_interval = int(data_len / total_log_times_num / batch_size / 100) * 100 * batch_size
    logger.info(f'proper_log_interval {proper_log_interval}')
    return proper_log_interval


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if self.warmup > 0  and epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
