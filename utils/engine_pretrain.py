import math
from typing import Iterable, Optional
import time
import torch
from timm.utils import accuracy
from . import misc
from . import lr_sched
import sys
import datetime

sys.path.append('../../')


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, scheduler=None,
                    args=None, logger=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 2

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if scheduler is None and data_iter_step % accum_iter == 0:

            lr_sched.adjust_learning_rate(optimizer=optimizer, epoch=data_iter_step / len(data_loader) + epoch,
                                          args=args)

        samples = samples.to(device, non_blocking=True).unsqueeze(1)
        targets = targets.to(device, non_blocking=True).to(torch.long)
        # print(samples, targets)
        # print(device)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            # for name, param in model.named_parameters():  # 返回网络的
            #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            #     log_writer.add_histogram(name + '_grad', param.grad, epoch_1000x)
            #     log_writer.add_histogram(name + '_data', param, epoch_1000x)
            optimizer.zero_grad()
        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if logger is not None:
                logger.log('loss', loss_value_reduce, epoch_1000x)
                logger.log('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if scheduler is not None:
        print(scheduler is not None)
        lr_sched.adjust_learning_rate(optimizer=optimizer, epoch=epoch, scheduler=scheduler,
                                      metric=metric_logger.meters['loss'].global_avg, args=args)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, By_Event):
    criterion = criterion

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 1, header):
        eeg = batch[0]
        target = batch[-1]
        eeg = eeg.to(device, non_blocking=True).unsqueeze(1)
        target = target.to(device, non_blocking=True).to(torch.long)
        # compute output

        with torch.cuda.amp.autocast():
            output = model(eeg)
            loss = criterion(output, target)
        start_time = time.time()
        if len(output.shape) == 3:
            output = torch.argmax(output, dim=1)
        by_event = By_Event(output, target.detach())
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Test: {} By Event Total time: {} )'.format(
            header, total_time_str))

        Recall, Precision, F1_score = by_event

        batch_size = eeg.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['Recall'].update(Recall, n=batch_size)
        metric_logger.meters['Precision'].update(Precision, n=batch_size)
        metric_logger.meters['F1_score'].update(F1_score, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Recall {recall.global_avg:.3f} Precision {precision.global_avg:.3f} F1_score {F1score.global_avg:.3f} '
          'loss {losses.global_avg:.3f} '
          .format(recall=metric_logger.Recall, precision=metric_logger.Precision, F1score=metric_logger.F1_score,
                  losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
