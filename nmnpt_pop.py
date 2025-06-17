
# tutorial population

import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF
from snntorch import utils
from snntorch import backprop
from nmnpt_mkdata import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# network parameters
num_inputs = 34*34
num_hidden = 128
# num_outputs = 10
num_steps = 1

# spiking neuron parameters
beta = 0.9  # neuron decay rate
grad = surrogate.fast_sigmoid()

pop_outputs = 500

# net
net_pop = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hidden),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                        nn.Linear(num_hidden, pop_outputs),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                        ).to(device)


# training 
loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=10)
optimizer = torch.optim.Adam(net_pop.parameters(), lr=2e-3, betas=(0.9, 0.999))

def test_accuracy(data_loader, net, num_steps, population_code=False, num_classes=False):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            utils.reset(net)
            spk_rec, _ = net(data)

            if population_code:
                acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=10) * spk_rec.size(1)
            else:
                acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)

            total += spk_rec.size(1)
    
    return acc/total

num_epochs = 5
# num_iters = 50

# training loop
for epoch in range(num_epochs):
    # for i, (data, targets) in enumerate(iter(trainloader)):
        # avg_loss = backprop.BPTT(net_pop, data, targets,
        #                          optimizer=optimizer, criterion=loss_fn, device=device)
    avg_loss = backprop.BPTT(net_pop, trainloader,
                        optimizer=optimizer, criterion=loss_fn,
                        time_var=True, time_first=True, device=device)


    print(f"Epoch: {epoch}")
    print(f"Test set accuracy: {test_accuracy(testloader, net_pop, num_steps, population_code=True, num_classes=10)*100:.3f}%\n")

        # if i == num_iters:
        #   break

