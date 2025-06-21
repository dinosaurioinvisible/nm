
# # tutorial for st-mnist

# # tonic imports
# import tonic
# import tonic.transforms as transforms  # Not to be mistaken with torchdata.transfroms
# from tonic import DiskCachedDataset

# # torch imports
# import torch
# from torch.utils.data import random_split
# from torch.utils.data import DataLoader
# import torchvision
# import torch.nn as nn

# # snntorch imports
# import snntorch as snn
# from snntorch import surrogate
# import snntorch.spikeplot as splt
# from snntorch import functional as SF
# from snntorch import utils

# # other imports
# import matplotlib.pyplot as plt
# from IPython.display import HTML
# from IPython.display import display
# import numpy as np
# import torchdata
# import os
# from ipywidgets import IntProgress
# import time
# import statistics

# from mkdata import SaccadeScramble
# both_sets = True
# time_window = 2000
# cx = 'seqs'

# dirpath = os.path.abspath(os.path.join(os.getcwd(),'..'))

# dataset = tonic.prototype.datasets.STMNIST(root=dirpath, keep_compressed = False, shuffle = False)

# sensor_size = tonic.prototype.datasets.STMNIST.sensor_size
# sensor_size = tuple(sensor_size.values())

# frame_transform = transforms.Compose([SaccadeScramble(),transforms.ToFrame(sensor_size=sensor_size, time_window=20000)])
# if both_sets:
#     frame_transform_test = transforms.Compose([SaccadeScramble(),transforms.ToFrame(sensor_size=sensor_size, time_window=20000)])
# else:
#     frame_transform_test = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size, time_window=20000)])

# def shorter_transform_STMNIST(data,transform_train,transform_test):
#     # short_train_size = 640
#     # short_test_size = 320
#     short_train_size = 1280
#     short_test_size = 640

#     train_bar = IntProgress(min=0, max=short_train_size)
#     test_bar = IntProgress(min=0, max=short_test_size)

#     testset = []
#     trainset = []

#     print('Porting over and transforming the trainset.')
#     display(train_bar)
#     # for _ in range(short_train_size):
#         # events, target = next(iter(data))
#     # make sequennces
#     for _ in range(0,short_train_size,2):
#         evs, tgt = next(iter(data))
#         sxe, sxt = next(iter(data))                  # for sequence as one input
#         events = np.pad(evs,(0,sxe.shape[0]))        # enlarge (new) for appending
#         sxe['t'] += 2000000                                 # real task time lapse
#         for vx in ['x','y','p','t']:                        # event params
#             events[vx] = np.concatenate((evs[vx],sxe[vx]))
#         target = [tgt, sxt]
#         # import pdb; pdb.set_trace()
#         events = transform_train(events)            # apply txs
#         trainset.append((events, target))           
#         train_bar.value += 1
#     print('Porting over and transforming the testset.')
#     display(test_bar)
#     # for _ in range(short_test_size):
#         # events, target = next(iter(data))
#     for _ in range(0,short_test_size,2):
#         evs, tgt = next(iter(data))         # TODO: target!
#         sxe, sxt = next(iter(data))
#         events = np.pad(evs,(0,sxe.shape[0]))
#         sxe['t'] += 2000000
#         for vx in ['x','y','p','t']:
#             events[vx] = np.concatenate((evs[vx],sxe[vx]))
#         target = [tgt, sxt]
#         # 
#         events = transform_test(events)
#         testset.append((events, target))
#         test_bar.value += 1

#     return (trainset, testset)

# start_time = time.time()
# trainset, testset = shorter_transform_STMNIST(dataset,frame_transform,frame_transform_test)
# elapsed_time = time.time() - start_time

# # Convert elapsed time to minutes, seconds, and milliseconds
# minutes, seconds = divmod(elapsed_time, 60)
# seconds, milliseconds = divmod(seconds, 1)
# milliseconds = round(milliseconds * 1000)

# # Print the elapsed time
# print(f"Elapsed time: {int(minutes)} minutes, {int(seconds)} seconds, {milliseconds} milliseconds")

# dataloader = DataLoader(trainset, batch_size=32, shuffle=True)

# transform = tonic.transforms.Compose([torch.from_numpy])

# cache_path_train = os.path.join(dirpath,'STMNIST/xcache/train_seq')
# cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path=cache_path_train)

# # no augmentations for the testset
# cache_path_test = os.path.join(dirpath,'STMNIST/xcache/test_seq')
# cached_testset = DiskCachedDataset(testset, cache_path=cache_path_test)

# batch_size = 32
# trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
# testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

# # Query the shape of a sample: time x batch x dimensions
# data_tensor, targets = next(iter(trainloader))
# print(data_tensor.shape)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # neuron and simulation parameters
# beta = 0.95

# # This is the same architecture that was used in the STMNIST Paper
# scnn_net = nn.Sequential(
#     nn.Conv2d(2, 32, kernel_size=4),
#     snn.Leaky(beta=beta, init_hidden=True),
#     nn.Conv2d(32, 64, kernel_size=3),
#     snn.Leaky(beta=beta, init_hidden=True),
#     nn.MaxPool2d(2),
#     nn.Flatten(),
#     nn.Linear(64 * 2 * 2, 10),  # Increased size of the linear layer
#     snn.Leaky(beta=beta, init_hidden=True, output=True)
# ).to(device)

# optimizer = torch.optim.Adam(scnn_net.parameters(), lr=2e-2, betas=(0.9, 0.999))
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

# def forward_pass(net, data):
#     spk_rec = []
#     utils.reset(net)  # resets hidden states for all LIF neurons in net

#     for step in range(data.size(0)):  # data.size(0) = number of time steps

#         spk_out, mem_out = net(data[step])
#         spk_rec.append(spk_out)

#     return torch.stack(spk_rec)

# start_time = time.time()

# num_epochs = 30

# loss_hist = []
# acc_hist = []

# # training loop
# for epoch in range(num_epochs):
#     for i, (data, targets) in enumerate(iter(trainloader)):
#         data = data.to(device)
#         targets = targets.to(device)

#         scnn_net.train()
#         spk_rec = forward_pass(scnn_net, data)
#         loss_val = loss_fn(spk_rec, targets)

#         # Gradient calculation + weight update
#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()

#         # Store loss history for future plotting
#         loss_hist.append(loss_val.item())

#         # Print loss every 4 iterations
#         if i%4 == 0:
#             print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
#             print(f"data shape: {data.shape}\n spikes: {spk_rec} \ntargets: {targets}")

#         # Calculate accuracy rate and then append it to accuracy history
#         acc = SF.accuracy_rate(spk_rec, targets)
#         acc_hist.append(acc)

#         # Print accuracy every 4 iterations
#         if i%4 == 0:
#             print(f"Accuracy: {acc * 100:.2f}%\n")

# end_time = time.time()

# # Calculate elapsed time
# elapsed_time = end_time - start_time

# # Convert elapsed time to minutes, seconds, and milliseconds
# minutes, seconds = divmod(elapsed_time, 60)
# seconds, milliseconds = divmod(seconds, 1)
# milliseconds = round(milliseconds * 1000)

# # Print the elapsed time
# print(f"Elapsed time: {int(minutes)} minutes, {int(seconds)} seconds, {milliseconds} milliseconds")

# # to save model
# # torch.save(scnn_net.state_dict(), 'scnn_net.pth')

# # Plot Loss
# fig = plt.figure(facecolor="w")
# plt.plot(acc_hist)
# plt.title("Train Set Accuracy")
# plt.xlabel("Iteration")
# plt.ylabel("Accuracy")
# plt.show()

# # Make sure your model is in evaluation mode
# scnn_net.eval()

# # Initialize variables to store predictions and ground truth labels
# acc_hist = []

# # Iterate over batches in the testloader
# with torch.no_grad():
#     for data, targets in testloader:
#         # Move data and targets to the device (GPU or CPU)
#         data = data.to(device)
#         targets = targets.to(device)

#         # Forward pass
#         spk_rec = forward_pass(scnn_net, data)

#         acc = SF.accuracy_rate(spk_rec, targets)
#         acc_hist.append(acc)

#         # if i%10 == 0:
#         # print(f"Accuracy: {acc * 100:.2f}%\n")
#         print(targets)

# print("The average loss across the testloader is:", statistics.mean(acc_hist))







