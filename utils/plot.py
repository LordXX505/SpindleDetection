import time
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import datasets



class DataPlot:

    def __init__(self) -> None:
        pass

    def __call__(self, dataset: datasets.MydataSet, nums: int, samples=5120, *args, **kwargs) -> None:
        self._plot(dataset, nums, samples)

    def get_param(self, nums) -> List[str]:
        color = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2"]
        return color[:nums]

    def get_random_index(self, subname: Optional[Union[List, int]] = None, nums: Optional[Union[List, int]] = None) -> List[np.ndarray]:
        if isinstance(subname, list):
            len_name = len(subname)
        elif isinstance(subname, int) or subname is None:
            len_name = subname
        else:
            raise Exception('subname is not list or int,recheck subname please')
        if isinstance(nums, list):
            n = len(nums)
        elif isinstance(nums, int) or nums is None:
            n = nums
        else:
            raise Exception('nums is not list or int,recheck nums please')
        res = []
        if len_name:
            subname_index = np.random.permutation(range(len_name))
            res.append(subname_index)
        if n:
            nums_index = np.random.permutation(range(n))
            res.append(nums_index)
        return res

    def _plot(self, dataset: datasets.MydataSet, nums: int, samples=5120) -> None:
        n = len(dataset)
        assert n >= nums
        index = self.get_random_index(nums=n)[0]
        print(index)
        # index = range(nums)
        while nums > 0:
            nrows = min(5, nums)
            color = self.get_param(nrows)
            fig, Axes = plt.subplots(nrows=nrows, ncols=1, sharex='all', figsize=(15, 8))
            fig.suptitle('RandomPlot')
            for i in range(nrows):
                axes = Axes[i]
                if isinstance(dataset, datasets.MydataSet):
                    data, label = dataset[index[nums - i - 1]]
                else:
                    data, label = dataset[0][index[nums - i - 1]], dataset[1][index[nums - i - 1]]
                print(nums - i - 1)
                mx = torch.max(data)
                mn = torch.min(data)
                axes.plot(range(samples), data, color=color[i])

                for x_st, x_ed in datasets.get_seg_label(label):
                    axes.fill_between(range(x_st, x_ed), mn, mx, facecolor='green', alpha=0.3)
                axes.set_xticks(np.arange(0, 5120, 256))
                axes.grid(True)
                # axes.legend()
            plt.savefig('../result/pits_{}'.format(nums))
            plt.clf()

            nums -= 5


if __name__ == '__main__':
    # sub_num1 = ['01', '02', '03', '04',
    #                                                             '05', '06', '07', '08', '09', '10', '11',
    #                                                             '12', '13', '14', '15', '16', '17', '18', '19']
    # # Expert2
    # sub_num2 = ['01', '02', '03',
    #            '05', '06', '07', '09', '10', '11',
    #            '12', '13', '14', '17', '18', '19']
    # dataset_val = build_datasets(args=None, is_train=False, sub_num=None)
    # dataset_train = build_datasets(args=None, is_train=True, sub_num=None)
    # print(len(dataset_val), len(dataset_train))
    dataset = datasets.build_datasets(args=None, is_train=False, sub_num=['01'])
    pt = DataPlot()
    pt(dataset, nums=10)
