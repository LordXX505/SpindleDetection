# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from torch.distributed.elastic.multiprocessing.errors import record
import argparse
import os
import random
import time
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import build_datasets
from utils.datasets import get_k_fold_index
import torch.utils.data
from utils import misc, get_optim
from utils.other import Fpfn
import SpindleUnet
from timm import optim
from utils.engine_pretrain import evaluate, train_one_epoch
import json
from utils.other import By_Event
import torch.distributed as dist
from utils.other import Logger
from utils.wandb import wandb_logger
def get_args_parser():
    parser = argparse.ArgumentParser('Main trainning for Spindle detection', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')

    # Model parameters
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout')
    parser.add_argument('--model', default='Unet_drop_fs4', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--Using_deep', default=False, action='store_true',
                        help='Using_deep or not (default: None)')
    parser.add_argument('--convs', default=3, type=int, help='Convolutions in one unet layer')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_k_fold', action='store_true', default=True)
    parser.add_argument('--threshold', default=0.55, type=int,
                        help='Softmax probability larger than threshold will be label 1')
    parser.add_argument('--IOU_threshold', default=0.2, type=int,
                        help='IOU threshold larger than the threshold is seen as a correct detection')
    # Criterion Strategy
    parser.add_argument('--Criterion', type=str, default="Fpfn", metavar='Criterion',
                        help='Criterion Strategy (default: None, else Fpfn)')
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--lr_policy', type=str, default="plateau",
                        help='learning rate adjustment policy (default is None)')
    parser.add_argument('--optim', type=str, default="adam",
                        help='learning rate adjustment policy')
    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--Augment', default=True, action='store_true',
                        help='Using augment or not')
    # Dataset parameters
    parser.add_argument('--batch_sampler', default=None, help='Batch sampler')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--datasets', default='MASS', type=str)
    parser.add_argument('--expert', default='E1', type=str, help='if you use MASS datasets, you should assign which '
                                                                 'annotation you want to ues')
    parser.add_argument('--output_dir', default='./checkpoint/2201210064/experiments',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../checkpoint/2201210064/experiments',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--pre_train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

@record
def main(args):
    misc.init_distributed_mode(args)
    # logger = Logger(os.path.join(args.output_dir))
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    device = torch.device(args.device)

    if args.datasets == 'MASS':
        if args.use_k_fold:
            train_val_index = get_k_fold_index(args, device, misc.get_rank())
            stats = {'Precision': 0.0, 'Recall': 0.0, 'F1_score': 0.0}
            k_index = 0
            for val_index, train_index in train_val_index:
                print(train_index, val_index)
                args.train_index = train_index
                args.val_index = val_index
                # logger = wandb_logger(args, args.job_id, k_index) if args.rank == 0 else None
                logger = None

                print('logger = %s' % str(logger))
                k_index += 1
                log_dir = os.path.join(args.log_dir, str(k_index))
                output_dir = os.path.join(args.output_dir, str(k_index))
                os.makedirs(log_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                dataset_train = build_datasets(args, True, sub_num=train_index)
                dataset_val = build_datasets(args, False, sub_num=val_index)
                if args.distributed:
                    num_tasks = misc.get_world_size()
                    global_rank = misc.get_rank()
                    if args.batch_sampler is None:
                        sampler_train = torch.utils.data.DistributedSampler(
                            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True
                        )
                    else:
                        raise NotImplementedError('batch_sampler')
                    assert sampler_train is not None
                    print("Sampler_train = %s" % str(sampler_train))
                    print("global_rank = %s" % str(global_rank))
                    if args.dist_eval:
                        if len(dataset_val) % num_tasks != 0:
                            print(
                                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                                'equal num of samples per-process.')
                        sampler_val = torch.utils.data.DistributedSampler(
                            dataset_val, num_replicas=num_tasks, rank=global_rank,
                            shuffle=True, drop_last=True)  # shuffle=True to reduce monitor bias
                    else:
                        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                else:
                    global_rank = 0
                    sampler_train = torch.utils.data.RandomSampler(dataset_train)
                    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

                if global_rank == 0 and args.log_dir is not None:
                    os.makedirs(log_dir, exist_ok=True)
                    log_writer = SummaryWriter(log_dir=log_dir)
                else:
                    log_writer = None

                data_loader_train = torch.utils.data.DataLoader(
                    dataset_train, sampler=sampler_train,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=True,
                )

                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val, sampler=sampler_val,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False
                )

                print(args.local_rank, SpindleUnet.__dict__)
                model = SpindleUnet.__dict__[args.model](
                    convs=args.convs,
                    Using_deep=args.Using_deep,
                    drop_out=args.dropout
                )
                model.to(device)

                model_without_ddp = model
                n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                n_parameters2 = sum(p.numel() for p in model.parameters() if p.grad is not None)
                print("Model = %s" % str(model_without_ddp))
                print('number of params (M): %.2f' % (n_parameters / 1.e6))
                print('number of params2 (M): %.2f' % (n_parameters2 / 1.e6))

                eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

                if args.lr is None:  # only base_lr is specified
                    args.lr = args.blr * eff_batch_size / 256

                print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
                print("actual lr: %.2e" % args.lr)

                print("accumulate grad iterations: %d" % args.accum_iter)
                print("effective batch size: %d" % eff_batch_size)
                if args.distributed:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                    model_without_ddp = model.module

                param_groups = optim.optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
                if args.optim is not None:
                    args.optim = "adam"
                optimizer = get_optim.get_optimizer(args, param_groups)

                # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

                if args.lr_policy is None:
                    # Using default
                    scheduler = None
                else:
                    scheduler = get_optim.get_scheduler(optimizer, args)
                print(scheduler)

                print(optimizer)
                loss_scaler = misc.NativeScalerWithGradNormCount()
                if args.Criterion is None or args.Criterion == 'crossentropyloss':
                    weight = torch.tensor([1.0, 4.0], device=device)
                    print('weight = ', weight)
                    criterion = torch.nn.CrossEntropyLoss(weight=weight)
                else:
                    criterion = Fpfn()
                print("criterion = %s" % str(criterion))
                misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler)

                By_Event_model = By_Event(args.threshold, args.IOU_threshold, device=device, freq=256, time=0.5)

                if args.eval:
                    test_stats = evaluate(data_loader_val, model, device, criterion, By_Event_model)
                    print(f"Precision of the network on the {len(dataset_val)} test images: {test_stats['Precision']:.1f}%")
                    exit(0)
                print(f"Start training for {args.epochs} epochs")

                start_time = time.time()
                max_Precision = 0.0
                last_loss = 0.0
                early_stop_count = 0

                if logger is not None:
                    logger.watch_model(model_without_ddp)
                for epoch in range(args.start_epoch, args.epochs):
                    if args.distributed:
                        data_loader_train.sampler.set_epoch(epoch)
                    train_stats = train_one_epoch(
                        model, criterion, data_loader_train,
                        optimizer, device, epoch, loss_scaler,
                        log_writer=log_writer, scheduler=scheduler,
                        args=args, logger=logger
                    )

                    if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
                        misc.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, output_dir=output_dir)

                    test_stats = evaluate(data_loader_val, model, device, criterion, By_Event_model)
                    print(f"Precision of the network on the {len(dataset_val)} test EEG: {test_stats['Precision']:.1f}%")
                    max_Precision = max(max_Precision, test_stats["Precision"])
                    print(f'Max accuracy: {max_Precision:.2f}%')
                    if log_writer is not None:
                        log_writer.add_scalar('perf/test_Precision', test_stats['Precision'], epoch)
                        log_writer.add_scalar('perf/test_Recall', test_stats['Recall'], epoch)
                        log_writer.add_scalar('perf/test_F1_score', test_stats['F1_score'], epoch)
                        for name, param in model.named_parameters():  # 返回网络的
                            log_writer.add_histogram(name + '_data', param, epoch)
                    if logger is not None:
                        logger.log('perf/test_Precision', test_stats['Precision'], epoch)
                        logger.log('perf/test_Recall', test_stats['Recall'], epoch)
                        logger.log('perf/test_F1_score',  test_stats['F1_score'], epoch)

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 **{f'test_{k}': v for k, v in test_stats.items()},
                                 'epoch': epoch,
                                 'expert:': args.expert,
                                 'n_parameters': n_parameters,
                                 'train_index': train_index,
                                 'val_index': val_index}
                    if args.output_dir and misc.is_main_process():
                        if log_writer is not None:
                            log_writer.flush()
                        with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n")
                    if epoch + 1 == args.epochs:
                        stats['Precision'] += test_stats['Precision']
                        stats['Recall'] += test_stats['Precision']
                        stats['F1_score'] += test_stats['F1_score']

                    if train_stats['loss'] == last_loss:
                        early_stop_count += 1
                        if early_stop_count == 20:
                            print('early stop: {}'.format(epoch))
                            break
                    else:
                        last_loss = train_stats['loss']
                        early_stop_count = 0
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Training time {}'.format(total_time_str))
            for k, v in stats.items():
                stats[k] = v/k_index
            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(stats) + "\n")
        else:
            raise NotImplementedError('use_k_fold')
    else:
        raise NotImplementedError('Using datasets MASS')


def add_graph(writer, args):
    img = torch.rand([1, 1, 5120], dtype=torch.float)
    model = SpindleUnet.__dict__[args.model](
                    convs=args.convs,
                    Using_deep=args.Using_deep,
                    drop_out=0
                )
    model.eval()

    out = model(img)
    print(out)
    with torch.no_grad():
        writer.add_graph(model, input_to_model=img)
    writer.close()




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    from pathlib import Path

    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # add_graph(writer=SummaryWriter(args.log_dir), args=args)

    main(args)
