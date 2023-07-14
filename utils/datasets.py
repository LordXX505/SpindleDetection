import os
import numpy.random as random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import numpy as np
from . import other, misc
from typing import Tuple, Any, List, Optional, Callable, Union


class MydataSet(Dataset):
    def __init__(self, transform: Optional[Callable], args, is_train: bool,
                 sub_num: Optional[Union[List[str], int]] = None) -> None:
        self.args = args
        self.transform = transform
        self.label_transform = torch.from_numpy
        self.data = []  # data path
        self.MASS_reject = ['04', '08', '15', '16']
        if is_train:
            real_path = os.path.join(args.data_path, 'train')
        else:
            real_path = os.path.join(args.data_path, 'val_quarter')

        if args.datasets == 'MASS':
            real_path = os.path.join(real_path, 'MASS_SS2')
        else:
            raise NotImplemented
        if args.Augment:
            if is_train:
                real_path = os.path.join(real_path, 'Aug')
        # Datasets == Mass

        if args.datasets == 'MASS':
            if sub_num is not None:
                for name in sub_num:
                    if args.expert == 'E2':
                        if name in self.MASS_reject:
                            raise Exception('Using Expert2, but doesnt exist the file :{}'.format(name))
                    sample = self._make_datasets_MASS(real_path, name)
                    self.data.extend(sample)
            else:
                sample = self._make_datasets_MASS(real_path)
                self.data.extend(sample)
        else:
            raise Exception('datasets must be MASS now')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_path, label_path = self.data[index]
        _data = np.load(data_path)
        _label = np.load(label_path)
        if self.label_transform is not None:
            _label = self.label_transform(_label)
            # _label = _label.int()
        if self.transform is not None:
            _data = self.transform(_data, self.args)
        # print(_data.dtype, _label.dtype)
        return _data, _label

    def __len__(self) -> int:
        return len(self.data)

    def _make_datasets_MASS(self, path: str, name: Optional[str] = None) -> List[Tuple[str, str]]:
        if name is not None:
            return make_datasets_MASS(self.args, os.path.join(path, name), reject=self.MASS_reject)
        else:
            return make_datasets_MASS(self.args, path, reject=self.MASS_reject)


def build_datasets(args: Optional[Union[argparse.Namespace, other.new_args]], is_train: bool,
                   sub_num: Optional[Union[List[str], int]] = None):
    if args is None:
        args = other.new_args(data_path='../data', expert='E1', model='SpindleU-Net', Augment=True, datasets='MASS')
        transform = build_transform(is_train)
        dataset = MydataSet(transform=transform, args=args, is_train=is_train, sub_num=sub_num)
        return dataset
    transform = build_transform(is_train)
    dataset = MydataSet(transform, args, is_train, sub_num, )

    print(dataset)

    return dataset


# prepare data but only using in augment
def data_pre_processing(data):
    for index, item in enumerate(data):
        for _, items in enumerate(item):
            if data[index][_] > 150:
                data[index][_] = 150
            if data[index][_] < -150:
                data[index][_] = -150
    max_item = np.max(data)
    min_item = np.min(data)
    for index, item in enumerate(data):
        for _, items in enumerate(item):
            data[index][_] = (data[index][_] - min_item) / (max_item - min_item) - 0.5
    return data


def train_transform(data: np.ndarray(shape=(5120,)), args: Optional[Union[argparse.Namespace, other.new_args]]) -> Optional[
    torch.Tensor]:
    if args.datasets == 'MASS':
        if args.Augment:
            res = data
        else:
            res = data_pre_processing(data)
    else:
        raise Exception('args.datasets != MASS')
    res = torch.from_numpy(res)
    res = res.float()
    return res


def val_transform(data: np.ndarray(shape=(5120,)), args: Optional[Union[argparse.Namespace, other.new_args]]) -> Optional[
    torch.Tensor]:
    if args.datasets == 'MASS':
        if args.Augment:
            res = data
        else:
            res = data_pre_processing(data)
    else:
        raise Exception('args.datasets != MASS')
    res = torch.from_numpy(res)
    res = res.float()

    return res


def build_transform(is_train: bool):
    if is_train:
        return train_transform
    else:
        return val_transform


def make_datasets_MASS(args: Optional[Union[argparse.Namespace, other.new_args]], path: str,
                       reject: Optional[List] = None) -> List[Tuple[str, str]]:
    instance = []
    for Path, Dirs, Files in sorted(os.walk(path)):
        filelist = os.listdir(Path)
        if 'data' in filelist and 'label' in filelist:
            # check if expert = E2, some files dont exist
            # print(os.path.basename(Path))
            # print(os.path.basename(os.path.dirname(Path)))
            if args.expert == 'E2' and str(os.path.basename(os.path.dirname(Path))) in reject:
                continue
            if str(os.path.basename(Path)) != args.expert:
                continue
            for _, dir, file in sorted(os.walk(os.path.join(Path, "data"))):
                for fname in sorted(file, key=lambda x: int(os.path.splitext(x)[0])):
                    num = os.path.splitext(fname)[0]
                    if args.expert == 'E1' or args.expert == 'E2':
                        label_path = os.path.join(Path, 'label')
                    else:
                        raise Exception('Using MASS datasets, but no expert or this parameters is wrong!')
                    item = os.path.join(_, str(num) + '.npy'), os.path.join(label_path,
                                                                            str(num) + '.npy')
                    instance.append(item)
        else:
            pass

    return instance


def get_k_fold_index(args: Optional[Union[argparse.Namespace, other.new_args]], device, rank):
    if args.datasets == 'MASS':
        if args.expert == 'E1':
            index = np.arange(1, 20)
        elif args.expert == 'E2':
            index = np.arange(1, 20)
            index = np.delete(index, [3, 7, 14, 15])
        else:
            raise Exception('Using MASS datasets, but no expert! or this parameters is wrong!')

        n = len(index)
        index = list(index)
        for i in range(n):
            if index[i] < 10:
                index[i] = '0' + str(index[i])
            else:
                index[i] = str(index[i])
        if rank == 0:
            np.random.shuffle(index)
        # print(index, rank)
        if misc.is_dist_avail_and_initialized():
            dist.broadcast_object_list(index, src=0)
        # print(index, rank)
        slice = n // 5
        for k in range(5):
            start = k * slice
            end = k * slice + slice
            yield index[start:end], index[0:start] + index[end:]
    else:
        raise Exception('Using MASS datasets, and you should not assign --datasets')



def get_seg_label(label: Optional[Union[np.ndarray, torch.Tensor]], samples=5120) -> Tuple[int, int]:
    i = 0
    while i < samples:
        if label[i] == 1:
            end = i
            for j in range(i, samples):
                end = j
                if label[j] != 1:
                    break
                if j == samples - 1:
                    end = samples
            yield i, end

            i = end
        else:
            i += 1


if __name__ == '__main__':
    data = build_datasets(args=None, is_train=False, sub_num=['01'])
    print(len(data))
    print(data[0])
    print(data[0])
    for val_index, train_index in get_k_fold_index(other.new_args(datasets='MASS', expert='E1'), device='cpu', rank=0):
        print(train_index)
        print(val_index)
