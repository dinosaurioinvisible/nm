
import os
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader


dirpath = os.path.abspath(os.path.join(os.getcwd(),'..','nmnist_ft_data'))

sensor_size = tonic.datasets.NMNIST.sensor_size
filter_time = 10000
time_window = 1000
frame_transform = transforms.Compose([transforms.Denoise(filter_time=filter_time),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=time_window)])

trainset = tonic.datasets.NMNIST(save_to=dirpath, transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to=dirpath, transform=frame_transform, train=False)

# cache_path_train = os.path.join(dirpath,'cache/nmnist/train')
# cached_trainset = DiskCachedDataset(trainset, cache_path=cache_path_train)
# cached_dataloader = DataLoader(cached_trainset)

# batch_size = 128
# train_loader = DataLoader(cached_trainset,
#                           batch_size=batch_size,
#                           collate_fn=tonic.collation.PadTensors())

# def load_sample_batched():
#     events, targets = next(iter(cached_dataloader))
#     return events, targets

import torch
import torchvision

pt_transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10,10])])

cache_path_train = os.path.join(dirpath,'cache/nmnist/train')
cached_trainset = DiskCachedDataset(trainset,
                                    transform=pt_transform,
                                    cache_path=cache_path_train)

# no augmentations for the testset
cache_path_test = os.path.join(dirpath,'cache/nmnist/test')
cached_testset = DiskCachedDataset(testset,
                                   cache_path=cache_path_test)

batch_size = 128
trainloader = DataLoader(cached_trainset,
                         batch_size=batch_size,
                         collate_fn=tonic.collation.PadTensors(batch_first=False),
                         shuffle=True)

testloader = DataLoader(cached_testset,
                        batch_size=batch_size,
                        collate_fn=tonic.collation.PadTensors(batch_first=False),
                        shuffle=True)

# event tensor: time-steps (padded), batch size, channels, height, width
# event_tensor, target = next(iter(trainloader))


