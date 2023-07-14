import math
from typing import Optional, Union
import argparse
from .other import new_args
import torch.optim.optimizer
import torch.optim.lr_scheduler


# default min_lr = 1e-6, warmup_epochs=5.
def adjust_learning_rate(optimizer = None,
                         scheduler=None,
                         epoch: Union[float, int] = None,
                         metric: Union[torch.Tensor] = None,
                         args: Optional[Union[argparse.Namespace, new_args]] = None) -> Union[torch.Tensor]:
    """Decay the learning rate with half-cycle cosine after warmup"""
    if scheduler is None:
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))  # cosine
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
    else:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics=metric)
        else:
            scheduler.step()
        lr = scheduler.optimizer.param_groups[0]['lr']
        return lr
