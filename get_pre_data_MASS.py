import mne
import numpy as np
import pandas as pd
import os
import re
import time
from typing import List
from utils.other import Event


# check the data
def check(Res_List, Spindle, PSG, Base, sub_num, threshold=0.25):
    segment = np.zeros(len(PSG))  # check if Spindle is right
    store_idx = []
    for index, item in enumerate(Res_List):
        if np.abs(item.first_time - Base[index]['onset']) > 0.01:
            raise Exception('Extract data from Sleep stage error!')
        if len(item.annotations):
            # make sure every epoch have at least one true spindle
            flag = 0
            if len(item.annotations) == 1:

                flag = 1
            for itemm in item.annotations:
                start_idx = PSG.time_as_index(itemm['onset'])[0]
                end_idx = PSG.time_as_index(itemm['onset'] + itemm['duration'])[0]
                if flag == 1:
                    lastitem_index = np.where(Spindle.onset <= itemm['onset'])[0]
                    lastitem = Spindle[lastitem_index[-1]]

                    if itemm['duration']/lastitem['duration'] >= threshold:
                        # print(itemm['duration'], lastitem['duration'])
                        store_idx.append(index)
                for i in range(start_idx, end_idx + 1):
                    segment[i] = 1
            if flag == 0:
                store_idx.append(index)
    # for i in store_idx:
    #     item = Res_List[i]
    #     print(item.first_time, item.first_time + 20)
    #     for itemm in item.annotations:
    #         lastitem_index = np.where(Spindle.onset <= itemm['onset'])[0]
    #
    #         lastitem = Spindle[lastitem_index[-1]]
    #
    #         if abs(itemm['duration'] - lastitem['duration']) > 0.001:
    #             print('0------------------------------------')
    #             print(len(item.annotations))
    #             print(itemm['onset'])
    #             print(itemm['duration'], lastitem['duration'])

    for item in Spindle:
        start_idx = PSG.time_as_index(item['onset'])[0]
        end_idx = PSG.time_as_index(item['onset'] + item['duration'])[0]
        for i in range(start_idx, end_idx + 1):  # must match
            if segment[i] == 0:
                print(i, segment[i])
                lastitem_index = np.where(Base.onset <= item['onset'])[0]
                if len(lastitem_index) != 0:
                    lastitem = Base[lastitem_index[-1]]
                    if lastitem['onset'] <= item['onset'] and item['onset'] + item['duration'] <= \
                            lastitem['onset'] + lastitem['duration']:
                        raise Exception(
                            'Extract data from Sleep stage error! Spindle doesnt match:{}'.format(item['onset']))
                    else:
                        print("this spindle is not all in stage2!:".format(item['onset']))
                        break
                else:
                    print("this spindle is not in stage2!:".format(item['onset']))
                    break
    return store_idx, True


def get_data_label(raw_list: List, store_idx: List, max_len=5120, amp=1000000):
    epochs = []
    label = []
    for _ in store_idx:
        item = raw_list[_]
        data = item.get_data().squeeze()
        epochs.append(amp * data[:max_len])
        annotations = item.annotations
        label.append(np.zeros(max_len))
        for annot in annotations:
            onset = annot["onset"] - item.first_time
            # be careful about near-zero errors (crop is very picky about this,
            # e.g., -1e-8 is an error)
            if -item.info['sfreq'] / 2 < onset < 0:
                onset = 0
            end = onset + annot["duration"]
            start_idx = item.time_as_index(onset)[0]  # return an array
            end_idx = min(max_len - 1, item.time_as_index(end)[0])  # this will lead length limit exceeded
            for i in range(start_idx, end_idx + 1):
                label[-1][i] = 1

    return epochs, label


def check_time(label):
    event_obj = Event(0, freq=1, time=1)
    for item in label:
        lf, rt, belong, group, unused = event_obj.get_event(np.array([item]), prob=False)
        if group[0] == 0:
            raise RuntimeError('group[0]==0')
        elif group[0] == 1:
            print(rt[0][1], lf[0][1], rt[0][1] - lf[0][1] + 1)


def train_run(file: str, Spindle_E1_path: str, Spindle_E2_path: str, Base_path: str, Root_path: str, sub_num: str):
    PSG = mne.io.read_raw_edf(os.path.join(Root_path, file), include=['EEG C3-CLE'])
    ana_path = Root_path.replace('bio', 'ana')
    Spindle_E1 = mne.read_annotations(os.path.join(ana_path, Spindle_E1_path))
    flag = 0
    if os.path.exists(os.path.join(ana_path, Spindle_E2_path)):
        flag = 1
    if flag:
        Spindle_E2 = mne.read_annotations(os.path.join(ana_path, Spindle_E2_path))
    Base = mne.read_annotations(os.path.join(ana_path, Base_path))
    drop_index = []
    for index, item in enumerate(Base):
        if item['description'] != 'Sleep stage 2':
            drop_index.append(index)
    Base.delete(drop_index)
    print("dropping Base index success :", len(drop_index))
    PSG_1 = PSG.copy()
    PSG_1.load_data()
    print('loading data')
    PSG_1.set_annotations(Spindle_E1)
    Res_List = PSG_1.crop_by_annotations(Base)
    sum_1 = len(Res_List)
    # return sum_1
    store_idx, check_flag = check(Res_List, Spindle_E1, PSG_1, Base, sub_num)
    if check_flag:
        print("Preparing data Spindle_1")

        epoch, label = get_data_label(Res_List, store_idx)
        epoch = np.array(epoch)
        label = np.array(label)
        sum_1 += epoch.shape[0] * epoch.shape[1]
        print(len(epoch), len(label), len(store_idx))
        os.makedirs('../SpindleData/E1', exist_ok=True)

        np.save('../SpindleData/E1/epoch_{}'.format(sub_num), epoch)
        np.save('../SpindleData/E1/label_{}'.format(sub_num), label)
    else:
        raise RuntimeError('loading data failed')
    if flag:
        print("starting {} Spindle E2".format(sub_num))
        PSG_2 = PSG.copy()
        PSG_2.load_data()
        print('loading data')
        PSG_2.set_annotations(Spindle_E2)
        Res_List = PSG_2.crop_by_annotations(Base)
        store_idx, check_flag = check(Res_List, Spindle_E2, PSG_2, Base, sub_num)
        if check_flag:
            print("Preparing data Spindle_2")
            epoch, label = get_data_label(Res_List, store_idx)
            epoch = np.array(epoch)
            label = np.array(label)
            print(len(epoch), len(label), len(store_idx))
            os.makedirs('../SpindleData/E2', exist_ok=True)

            np.save('../SpindleData/E2/label_{}'.format(sub_num), label)
            np.save('../SpindleData/E2/epoch_{}'.format(sub_num), epoch)
        else:
            raise RuntimeError('loading data failed')
        print('loading data success, idx:{}'.format(sub_num))


def val_run(file: str, Spindle_E1_path: str, Spindle_E2_path: str, Base_path: str, Root_path: str, sub_num: str):
    PSG = mne.io.read_raw_edf(os.path.join(Root_path, file), include=['EEG C3-CLE'])
    ana_path = Root_path.replace('bio', 'ana')
    Spindle_E1 = mne.read_annotations(os.path.join(ana_path, Spindle_E1_path))
    flag = 0
    if os.path.exists(os.path.join(ana_path, Spindle_E2_path)):
        flag = 1
    if flag:
        Spindle_E2 = mne.read_annotations(os.path.join(ana_path, Spindle_E2_path))
    Base = mne.read_annotations(os.path.join(ana_path, Base_path))
    drop_index = []
    for index, item in enumerate(Base):
        if item['description'] != 'Sleep stage 2':
            drop_index.append(index)
    Base.delete(drop_index)
    print("dropping Base index success :", len(drop_index))
    PSG_1 = PSG.copy()
    PSG_1.load_data()
    print('val_run: loading data')
    PSG_1.set_annotations(Spindle_E1)
    Res_List = PSG_1.crop_by_annotations(Base)
    sum_1 = len(Res_List)
    # return sum_1
    store_idx, check_flag = check(Res_List, Spindle_E1, PSG_1, Base, sub_num)
    if check_flag:
        print("Preparing data Spindle_1")
        # store_idx = [i for i in range(0, len(Res_List))]
        epoch, label = get_data_label(Res_List, store_idx)
        epoch = np.array(epoch)
        label = np.array(label)
        # sum_1 += epoch.shape[0] * epoch.shape[1]
        print(len(epoch), len(label), len(store_idx))
        os.makedirs('../SpindleData/val_quarter/E1', exist_ok=True)

        np.save('../SpindleData/val_quarter/E1/epoch_{}'.format(sub_num), epoch)
        np.save('../SpindleData/val_quarter/E1/label_{}'.format(sub_num), label)
    else:
        raise RuntimeError('loading data failed')
    if flag:
        print("starting {} Spindle E2".format(sub_num))
        PSG_2 = PSG.copy()
        PSG_2.load_data()
        print('loading data')
        PSG_2.set_annotations(Spindle_E2)
        Res_List = PSG_2.crop_by_annotations(Base)
        store_idx, check_flag = check(Res_List, Spindle_E2, PSG_2, Base, sub_num)
        if check_flag:
            print("Preparing data Spindle_2")
            # store_idx = [i for i in range(0, len(Res_List))]

            epoch, label = get_data_label(Res_List, store_idx)
            epoch = np.array(epoch)
            label = np.array(label)
            print(len(epoch), len(label), len(store_idx))
            os.makedirs('../SpindleData/val_quarter/E2', exist_ok=True)

            np.save('../SpindleData/val_quarter/E2/label_{}'.format(sub_num), label)
            np.save('../SpindleData/val_quarter/E2/epoch_{}'.format(sub_num), epoch)
        else:
            raise RuntimeError('loading data failed')
        print('loading data success, idx:{}'.format(sub_num))
    return sum_1

def get_annotations(file: str, Spindle_E1_path: str, Spindle_E2_path: str, Base_path: str,
                    Root_path: str, sub_num: str, Index=None):
    PSG = mne.io.read_raw_edf(os.path.join(Root_path, file), include=['EEG C3-CLE'])
    ana_path = Root_path.replace('bio', 'ana')
    Spindle_E1 = mne.read_annotations(os.path.join(ana_path, Spindle_E1_path))
    flag = 0
    if os.path.exists(os.path.join(ana_path, Spindle_E2_path)):
        flag = 1
    if flag:
        Spindle_E2 = mne.read_annotations(os.path.join(ana_path, Spindle_E2_path))
    Base = mne.read_annotations(os.path.join(ana_path, Base_path))
    drop_index = []
    for index, item in enumerate(Base):
        if item['description'] != 'Sleep stage 2':
            drop_index.append(index)
    Base.delete(drop_index)
    PSG.set_annotations(Spindle_E1)
    # print(Spindle_E2.onset[679])
    # print(Spindle_E2.onset[680], Spindle_E2.duration[680])
    # print(Spindle_E2.onset[681], Spindle_E2.duration[681])
    Res_List = PSG.crop_by_annotations(Base)
    for i in Index:
        n = len(Res_List)
        item = Res_List[i%n]
        print('----------Index----------')
        print(item.first_time)
        if item.annotations:
            for itemm in item.annotations:
                print('---------- ', i, ' ----------')
                print('onset: ', itemm["onset"])
                print('duration: ', itemm["duration"])
    print(Base.onset[:10])
    # print(Spindle_E2.onset[:20], Spindle_E2.duration[:20])
    dur_min = np.min(Spindle_E1.duration)
    dur_max = np.max(Spindle_E1.duration)
    print('E1:', dur_min, dur_max)
    if flag:
        dur_min = np.min(Spindle_E2.duration)
        dur_max = np.max(Spindle_E2.duration)
        print('E2:', dur_min, dur_max)


def main():
    Root_path = "../data/SS2/SS2_bio"
    sum_val = 0
    sum_train = 0
    for path, subdirs, files in os.walk(Root_path):

        for file in files:
            if file.endswith('PSG.edf'):
                start_time = time.time()
                sub_num = file[8:10]
                sub_num = ['01', '02', '03', '04',
                 '05', '06', '07', '08', '09', '10', '11',
                 '12', '13', '14', '15', '16', '17', '18', '19']
                sub_num = sub_num[0]
                print("starting data extract: ", sub_num)
                Spindle_E1_path = '01-02-00{} Spindles_E1.edf'.format(sub_num)
                Spindle_E2_path = '01-02-00{} Spindles_E2.edf'.format(sub_num)
                Base_path = '01-02-00{} Base.edf'.format(sub_num)
                log_msg = [
                    'sub_num: ' + sub_num,
                    'Spindle_E1: ' + Spindle_E1_path,
                    'Spindle_E2: ' + Spindle_E2_path,
                    'Base: ' + Base_path,
                    'time: {time}'
                ]
                print('\t'.join(log_msg).format(time=start_time))
                # train_run(file, Spindle_E1_path, Spindle_E2_path, Base_path, Root_path, sub_num)
                # sum_val += val_run(file, Spindle_E1_path, Spindle_E2_path, Base_path, Root_path, sub_num)
                # sum_val += val_run(file, Spindle_E1_path, Spindle_E2_path, Base_path, Root_path, sub_num)
                # sum_train += train_run(file, Spindle_E1_path, Spindle_E2_path, Base_path, Root_path, sub_num)
                get_annotations(file, Spindle_E1_path, Spindle_E2_path, Base_path, Root_path, sub_num,
                                Index=[428 ,408 ,460, 296 ,499 ,417  ,21 ,426 ,468 ,215])
                exit(0)
                end_time = time.time()
                print("time cost = ", end_time - start_time)
    print("All success")
    print(sum_train, sum_val)


if __name__ == "__main__":
    main()
