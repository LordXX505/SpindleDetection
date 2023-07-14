import os
import re

import numpy as np
import random
from datasets import data_pre_processing
from other import Event


def SpindleAug(data: np.ndarray, label: np.ndarray,  mu=0, sigma=1):
    Auglabel = label.copy()
    Augdata = data.copy()
    Augdata += np.random.normal(mu, sigma, len(data))
    Augdata *= np.random.normal(1, 0.01, len(data))
    rd = np.random.rand(1)
    if rd > 0.5:
        Augdata = Augdata[::-1]
        Auglabel = Auglabel[::-1]
    return Augdata, Auglabel


def Unet_Aug(Datapath, sub_num, ResPath, expert, mu=0, sigma=1) -> None:
    MASS_reject = ['04', '08', '15', '16']

    for name in sub_num:
        if expert == 'E2':
            if name in MASS_reject:
                continue
        print("starting to split raw data. name = ", name)
        expert_path = os.path.join(Datapath, expert)
        path = os.path.join(expert_path, 'epoch_' + name + '.npy')
        data = np.load(path)
        label_path = os.path.join(expert_path, 'label_' + name + '.npy')
        label = np.load(label_path)

        newdata = []
        newlabel = []

        for index, item in enumerate(data):
            Augdata, Auglabel = SpindleAug(item, label[index])
            # print(Augdata, item)
            # print(Auglabel, label[index])
            newdata.append(Augdata)
            newlabel.append(Auglabel)
        newdata = np.array(newdata)
        newlabel = np.array(newlabel)
        data = np.concatenate((data, newdata), axis=0)
        label = np.concatenate((label, newlabel), axis=0)
        data = data_pre_processing(data)
        print('saving' + expert + ' ' + name + '..................')
        # save Augment
        datapath = os.path.join(ResPath, 'MASS_SS2')
        datapath = os.path.join(datapath, 'Aug')
        res_path = os.path.join(datapath, name)
        os.makedirs(os.path.join(os.path.join(res_path, expert), 'data'), exist_ok=True)
        os.makedirs(os.path.join(os.path.join(res_path, expert), 'label'), exist_ok=True)
        for _, item in enumerate(data):
            _path = os.path.join(os.path.join(os.path.join(res_path, expert), 'data'), str(_))
            np.save(_path, item)
            _path = os.path.join(os.path.join(os.path.join(res_path, expert), 'label'), str(_))
            np.save(_path, label[_])
        print('{} over'.format(name))


def get_val(Datapath, sub_num, ResPath, expert):
    MASS_reject = ['04', '08', '15', '16']

    for name in sub_num:
        if expert == 'E2':
            if name in MASS_reject:
                continue
        print("starting to split raw data. name = ", name)
        expert_path = os.path.join(Datapath, expert)
        path = os.path.join(expert_path, 'epoch_' + name + '.npy')
        data = np.load(path)
        label_path = os.path.join(expert_path, 'label_' + name + '.npy')
        label = np.load(label_path)

        data = data_pre_processing(data)
        print('val saving..................')
        datapath = os.path.join(ResPath, 'MASS_SS2')
        res_path = os.path.join(datapath, name)
        os.makedirs(os.path.join(os.path.join(res_path, expert), 'data'), exist_ok=True)
        os.makedirs(os.path.join(os.path.join(res_path, expert), 'label'), exist_ok=True)
        for _, item in enumerate(data):
            _path = os.path.join(os.path.join(os.path.join(res_path, expert), 'data'), str(_))
            np.save(_path, item)
            _path = os.path.join(os.path.join(os.path.join(res_path, expert), 'label'), str(_))
            np.save(_path, label[_])
        print('{} over'.format(name))


def Spindel_PreProcessing(expert, sub_num, Datapath=os.path.join('../', 'data/train/Aug/SS2'), sample_len =5120, flag=1):
    Path = os.path.join('../', 'Spindle_unet_data/train/Aug/SS2')
    os.makedirs(Path, exist_ok=True)
    MASS_reject = ['04', '08', '15', '16']

    for name in sub_num:
        if expert == 'E2':
            if name in MASS_reject:
                continue
        path = os.path.join(Datapath, name)
        label_path = os.path.join(path, "label")
        count = []
        event = Event(0)
        for _, subdirs, files in sorted(os.walk(label_path)):
            file_lf = {}
            file_rt = {}
            file_belong = {}
            file_group = {}
            for file in files:
                ep = re.findall(pattern=r'E[0-9]', string=file)
                number = re.findall(pattern=r'[0-9]*_', string=file)[0][:-1]
                if ep[0] != expert:
                    continue
                number = int(number)
                count.append(number)
                label = np.load(os.path.join(_, file))
                index = 0
                for item in label:
                    if int(item) == 1:
                        index += 1
                lf, rt, belong, group, unused = event.get_event(np.array([label]), prob=False)
                file_lf[number] = lf.squeeze()
                file_rt[number] = rt.squeeze()
                file_belong[number] = belong.squeeze()
                file_group[number] = group
            store = []
            print(sorted(count))
            for i in range(len(file_group)):
                if file_group[i][0] == 0:
                    continue
                elif file_group[i][0] == 1:
                    if file_lf[i][1].item() == 0:
                        if i == 0:
                            continue
                        else:
                            print('file_rt:', file_rt[i][1])
                            print('file_lf:', file_lf[i][1])
                            print(file_rt[i-1], file_lf[i-1])
                            if file_rt[i-1][-1].item() == sample_len-1:
                                len1 = file_rt[i][1] - file_lf[i][1] + 1
                                len2 = file_rt[i-1][-1] - file_lf[i-1][-1] + 1
                                if (1.0*len1)/(len1+len2) >= 0.25:
                                    store.append(i)
                                else:
                                    continue
                            else:
                                continue
                    elif file_rt[i][1].item() == sample_len-1:
                        print('file_rt:', file_rt[i][1])
                        print('file_lf:', file_lf[i][1])
                        if i != len(file_group)-1 and file_lf[i + 1][1].item() == 0:
                            print(file_rt[i + 1][1], file_lf[i + 1][1])
                            len1 = file_rt[i][1] - file_lf[i][1] + 1
                            len2 = file_rt[i + 1][1] - file_lf[i + 1][1] + 1
                            if (1.0 * len1) / (len1 + len2) >= 0.25:
                                store.append(i)
                            else:
                                continue
                        else:
                            continue
                else:
                    store.append(i)
            label_store_path = os.path.join(os.path.join(os.path.join(Path, name), 'label'), expert)

            os.makedirs(label_store_path, exist_ok=True)
            for number in store:
                file = str(number) + '_' + expert + '.npy'
                label = np.load(os.path.join(_, file))
                np.save(file, label)
                if flag == 1:
                    file = os.path.join(os.path.join(os.path.join(Datapath, name), 'data'), str(number) + '.npy')
                    data = np.load(file)
                    np.save(file, data)


if __name__ == '__main__':
    # Spindel_PreProcessing(expert='E1', sub_num=['01'])

    # Spindel_PreProcessing(expert='E1', sub_num=['01', '02', '03', '04', '05', '06', '07', '08', '09',
    #                                             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
    # print('starting')
    Unet_Aug(Datapath='../SpindleData', sub_num=['01', '02', '03', '04',
                                                     '05', '06', '07', '08', '09', '10', '11',
                                                     '12', '13', '14', '15', '16', '17', '18', '19'],
              ResPath='../data/train', expert='E1')
    Unet_Aug(Datapath='../SpindleData', sub_num=['01', '02', '03', '04',
                                                 '05', '06', '07', '08', '09', '10', '11',
                                                 '12', '13', '14', '15', '16', '17', '18', '19'],
             ResPath='../data/train', expert='E2')
    get_val(Datapath='../SpindleData/val_quarter', sub_num=['01', '02', '03', '04',
                                                    '05', '06', '07', '08', '09', '10', '11',
                                                    '12', '13', '14', '15', '16', '17', '18', '19'],
             ResPath='../data/val_quarter',  expert='E1')
    get_val(Datapath='../SpindleData/val_quarter', sub_num=['01', '02', '03', '04',
                                                    '05', '06', '07', '08', '09', '10', '11',
                                                    '12', '13', '14', '15', '16', '17', '18', '19'],
            ResPath='../data/val_quarter', expert='E2')

