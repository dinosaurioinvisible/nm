
from pdb import set_trace as pp
import numpy as np
import matplotlib.pyplot as plt
from mkdata import mk_custom_loaders
import tonic

trs, trs_lbs, ts, ts_lbs = mk_custom_loaders(train_size=1000,
                                test_size=100,
                                torch_dl=False,
                                seq_test=False,
                                sep_labels=True)

trs_lbs, ts_lbs = np.array(trs_lbs), np.array(ts_lbs)

# 1. time difs examples
# for i in range(3):
#     vt = ts[i]['t']
#     vdt = vt[:-1] - vt[1:]
#     plt.plot(np.abs(vdt))
# plt.show()

# for simplicity
twindow=10000
tx_tf = tonic.transforms.ToFrame(
    sensor_size = tonic.datasets.NMNIST.sensor_size,
    time_window=twindow)

# 2. mk num case tensor

# (assuming cont equal n of data-label cases)
def mk_num_tensor(dset, lbs, num=3,
                             neg=False,
                             nframes=30,
                             max_size=128):
    # define tensor
    ncases = int(len(lbs)/10) if len(lbs) <= max_size*10 else max_size
    txe = np.zeros((ncases,nframes,34,34))
    # get evs data
    pol = 1 if neg == False else 0          # for dif channels
    lb_ids = np.where(lbs==num)[0]          # num cases ids
    for ei,lb_id in enumerate(lb_ids):
        evs = dset[lb_id]                   # events
        evs = evs[np.where(evs['p']==pol)]
        fr_evs = tx_tf(evs)                 # frames, shape: ~ 30, 2, 34, 34
        fr_evs = fr_evs[:,pol]              # reduce to: 30, 34, 34
        txe[ei,:fr_evs.shape[0]] = fr_evs   # 'pad' to 30
    return txe

txe = mk_num_tensor(ts,ts_lbs,num=3,neg=False)      # ncases, 30, 34, 34

# 3. stacking tensor values 
# opt1 - sum corr frames to make a 30, 34, 34 tensor 
txsum = np.zeros((30,34,34))
for i in range(30):
    txsum[i] = txe[:,i].sum(axis=0) 
ani = tonic.utils.plot_animation(txsum)
# opt2 - sum everything into a 34,34 matrix (like in the article)
# tsc = txe.sum(axis=(0,1))

# 4. map tensor to CA grid 
# x,y: [0,34]
ca = np.zeros((30,34*2,34*2))
cmn = 2    # CA size MxM 68x68/ P tensor size NxN 34x34 
k = 17
kr = int((k-1)/2)
mask = np.zeros((k,k))
for i in range(k):
    mask[i:k-i,i:k-i] = 1/2 ** ((k-1)/2-i)  # t / 2 ** (8-i)

for (f,i,j) in np.array(txsum.nonzero()).T:   # non zero pixels
    t = txsum[f,i,j]    # temperature
    xo,yo = (np.array([i,j]) * cmn - kr).clip(0,68).astype(int) # i,j -> x,y
    xf,yf = (np.array([i,j]) * cmn + kr +1).clip(0,68).astype(int)
    mio = 0 if xo > 0 else k - (xf-xo)
    mjo = 0 if yo > 0 else k - (yf-yo)
    mif = k if xf <= 60 else xf-xo
    mjf = k if yf <= 60 else yf-yo
    ca[f,xo:xf,yo:yf] += mask[mio:mif,mjo:mjf] * t    # CA ROI
    
ani = tonic.utils.plot_animation(ca)    # 30, 68, 68

# 5. ca for frames?





