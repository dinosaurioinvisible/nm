import os
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
from pdb import set_trace as pp
import numpy as np

def mk_custom_loaders(dataset='nmnist',
                      train_size=1200,
                      test_size=600,
                      torch_dl=True,
                      seq_test=False,
                      sep_labels=False):

    dirpath = os.path.abspath(os.path.join(os.getcwd(),'..','{}_data'.format(dataset)))

    time_window = 10000
    sensor_size = tonic.datasets.NMNIST.sensor_size
    tx_tf = tonic.transforms.ToFrame(sensor_size=sensor_size,
                                    time_window=time_window)

    dir_txs = 'seq'
    tx_path = os.path.join(dirpath,dir_txs)
    train_dataset = tonic.datasets.NMNIST(save_to=tx_path, train=True)
    test_dataset = tonic.datasets.NMNIST(save_to=tx_path, train=False)

    train_size = train_size
    test_size = test_size
    train_step = int(len(train_dataset)/train_size)
    test_step = int(len(test_dataset)/test_size)
    trainset, testset, testset_seq = [], [], []
    trainset_lbs, testset_lbs, testset_seq_lbs = [], [], []

    for ti in range(0,train_size):
        txi = ti * train_step
        ev,tg = train_dataset[txi]
        if torch_dl:
            ev = tx_tf(ev)
            trainset.append((ev,tg))
        elif sep_labels:
            trainset.append(ev)
            trainset_lbs.append(tg)
        else:
            trainset.append((ev,tg))

    for ti in range(0,test_size):
        txi = ti * test_step
        ev,tg = test_dataset[txi]
        if torch_dl:
            ev = tx_tf(ev)
            testset.append((ev,tg))
        elif sep_labels:
            testset.append(ev)
            testset_lbs.append(tg)
        else:
            testset.append((ev,tg))

    if seq_test:
        for ti in range(0,test_size):
            ri1,ri2,ri3 = np.random.randint(0,10,size=3) * 1000 + ti
            ev1,tg1 = test_dataset[ri1]
            ev2,tg2 = test_dataset[ri2]
            ev3,tg3 = test_dataset[ri3]
            evj = np.pad(ev1, (0, ev2.shape[0]+ev3.shape[0]))
            ev2['t'] += ev1['t'][-1]
            ev3['t'] += ev2['t'][-1]
            for vi in ['x','y','p','t']:
                evj[vi] = np.hstack((ev1[vi],ev2[vi],ev3[vi]))
            if torch_dl:
                evj = tx_tf(evj)
                testset_seq.append((evj,[tg1,tg2,tg3]))
            elif sep_labels:
                testset_seq.append(evj)
                testset_seq_lbs.append([tg1,tg2,tg3])
            else:
                testset_seq.append((evj,[tg1,tg2,tg3]))

    if torch_dl:
        trainloader = DataLoader(trainset)
        testloader = DataLoader(testset)
        # txe.size: 1,30,2,34,34 <:> torch.Size([89, 32, 2, 10, 10])
        # frames, targets = next(iter(trainloader))
        if seq_test:
            testloader_seq = DataLoader(testset_seq)
            return trainloader, testloader, testloader_seq
        return trainloader, testloader
    
    if seq_test:
        if sep_labels:
            return trainset, trainset_lbs, testset, testset_lbs, testset_seq, testset_seq_lbs
        return trainset, testset, testset_seq
    if sep_labels:
        return trainset, trainset_lbs, testset, testset_lbs
    return trainset, testset


def mk_loaders(dataset='nmnist',
               both_sets=True,
               batch_size=128, 
               shuffle_train=True, shuffle_test=True, 
               twindow=1000, tfilter=10000,
               tjitter=False, treversal=False, mergep=False, scramble=False):
    
    dirpath = os.path.abspath(os.path.join(os.getcwd(),'..','{}_data'.format(dataset)))

    sensor_size = tonic.datasets.NMNIST.sensor_size

    txs = []
    dir_txs = 'txs_'
    if tfilter > 0:
        denoise_tx = transforms.Denoise(filter_time = tfilter)
        dir_txs += 'd'
        txs.append(denoise_tx)
    if tjitter:
        time_jitter_tx = transforms.TimeJitter(std = 100, clip_negative=True)
        dir_txs += 'j'
        txs.append(time_jitter_tx)
    if treversal:
        rt_rev_tx = transforms.RandomTimeReversal(p = 1, flip_polarities=False)
        dir_txs += 'r'
        txs.append(rt_rev_tx)
    if mergep: 
        merge_pols_tx = transforms.MergePolarities()
        sensor_size = (34,34,1)
        dir_txs += 'm'
        txs.append(merge_pols_tx)
    if scramble:
        scramble_tx = SaccadeScramble()
        dir_txs += 'x'
        txs.append(scramble_tx)
    if twindow > 0:
        frame_tx = transforms.ToFrame(sensor_size=sensor_size, time_window=twindow)
        dir_txs += 'w'
        txs.append(frame_tx)
    
    frame_transform = transforms.Compose(txs)
    tx_path = os.path.join(dirpath,dir_txs)

    trainset = tonic.datasets.NMNIST(save_to=tx_path, transform=frame_transform, train=True)
    if both_sets:
        testset = tonic.datasets.NMNIST(save_to=tx_path, transform=frame_transform, train=False)
    else:
        transform_test = transforms.Compose([transforms.Denoise(filter_time=10000),
                                             transforms.ToFrame(sensor_size=sensor_size,
                                                                time_window=1000)])
        testset = tonic.datasets.NMNIST(save_to=tx_path, transform=transform_test, train=False)

    cache_path_train = os.path.join(tx_path,'cache/train')
    cached_trainset = DiskCachedDataset(trainset, cache_path=cache_path_train)

    cache_path_test = os.path.join(tx_path,'cache/test')
    cached_testset = DiskCachedDataset(testset, cache_path=cache_path_test)

    trainloader = DataLoader(cached_trainset,
                            batch_size=batch_size,
                            collate_fn=tonic.collation.PadTensors(batch_first=False),
                            shuffle=shuffle_train)

    testloader = DataLoader(cached_testset,
                            batch_size=batch_size,
                            collate_fn=tonic.collation.PadTensors(batch_first=False),
                            shuffle=shuffle_test)

    return trainloader, testloader



class EventScramble:
    def __call__(self, events):
        events = events.copy()
        pols = [0,1]
        for pol in pols:
            locs = np.where(events['p']==pol)[0]
            rlocs = np.copy(locs)
            np.random.shuffle(rlocs)
            events['x'][locs] = events['x'][rlocs]
            events['y'][locs] = events['y'][rlocs]
        return events
