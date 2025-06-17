
# tutorial snn regression 2 
# recurrent leaky integrate-and-fire neurons
# convolutional feedback (to map from 3d outputs into a linear layer)

import snntorch as snn
from snntorch import functional as SF

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import tqdm
from mkdata import mk_loaders


train_loader, test_loader = mk_loaders(mergep=True)

class Net(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, hidden, beta):
        super().__init__()

        self.timesteps = timesteps
        self.hidden = hidden
        self.beta = beta

        # layer 1
        self.fc1 = torch.nn.Linear(in_features=1156, out_features=self.hidden)  # input map to layer: 34 * 34
        self.rlif1 = snn.RLeaky(beta=self.beta, linear_features=self.hidden)    # rlif layer 1

        # layer 2
        self.fc2 = torch.nn.Linear(in_features=self.hidden, out_features=10)    # hidden to rlif2
        self.rlif2 = snn.RLeaky(beta=self.beta, linear_features=10)             # rlif2 -> output

    def forward(self, x):
        """Forward pass for several time steps."""

        # Initalize membrane potential
        spk1, mem1 = self.rlif1.init_rleaky()       # outputs spikes from elements in batch
        spk2, mem2 = self.rlif2.init_rleaky()       # memb pot for each element (tensor from batch)

        # Empty lists to record outputs
        spk_recording = []
        mem_recording = []

        # for step in range(self.timesteps):
        for step in range(x.size(0)):
            spk1, mem1 = self.rlif1(self.fc1(x[step]), spk1, mem1)
            spk2, mem2 = self.rlif2(self.fc2(spk1), spk2, mem2)
            spk_recording.append(spk2)
            mem_recording.append(mem2)

        return torch.stack(spk_recording)
        # return torch.stack(mem_recording)


hidden = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
num_steps = 10
beta = 0.9
model = Net(timesteps=num_steps, hidden=hidden, beta=beta).to(device)

# training 

loss_function = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
# loss_function = SF.mse_membrane_loss(on_target=1.05, off_target=0.2)


num_epochs = 3
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_hist = []
num_iters = 50

with tqdm.trange(num_epochs) as pbar:
    for _ in pbar:
        # train_batch = iter(train_loader)
        # minibatch_counter = 0
        loss_epoch = []
        for i, (data, targets) in enumerate(iter(train_loader)):
            # for feature, label in train_batch:
            minibatch_counter = 0

            targets = targets.to(device)            # 128 targets (for each case)
            data = data[:,:,0,...]                  # reshape to ~300, 128, 34, 34
            data = data.to(device)

            spks_out = []
            for bi in range(128):                   # for each case/number-sampling
                feature = data[:,bi,...]            # ~300 timesteps
                # feature = feature.to(device)
                spk = model(feature.flatten(1))     # forward pass, returns (300,10)
                # mem = model(feature.flatten(1))   
                spk = spk.sum(axis=0)
                spks_out.append(spk)
            spks_out = torch.stack(spks_out)
            # import pdb; pdb.set_trace()
            loss_val = loss_function(spks_out, targets) # apply loss
            optimizer.zero_grad() # zero out gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights

            loss_hist.append(loss_val.item())

            minibatch_counter += 1

            avg_batch_loss = sum(loss_hist) / minibatch_counter
            pbar.set_postfix(loss="%.3e" % avg_batch_loss)

            if i == num_iters:
                break


test_batch = iter(test_loader)
# minibatch_counter = 0
# loss_epoch = []

model.eval()
with torch.no_grad():
    total = 0
    acc = 0
    for feature, label in test_batch:
        feature = feature[:,:,0,...]
        feature = feature.to(device)
        label = label.to(device)
        # spk = model(feature.flatten(1)) # forward-pass
        spk = model(feature.flatten(2))
        print(spk)
        print(label)
        acc += SF.accuracy_rate(spk, label) * spk.size(1)
        total += spk.size(1)


print(f"The total accuracy on the test set is: {(acc/total) * 100:.2f}%")

