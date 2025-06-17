
from pdb import set_trace as pp
import numpy as np
import matplotlib.pyplot as plt
from mkdata import mk_custom_loaders
from timeColouredPlots import doColourVaryingPlot2d as colorplot
import tonic

trs, trs_lbs, ts, ts_lbs = mk_custom_loaders(train_size=1000,
                                test_size=100,
                                torch_dl=False,
                                seq_test=False,
                                sep_labels=True)

trs_lbs, ts_lbs = np.array(trs_lbs), np.array(ts_lbs)

# visualize saccade order
vi = ts[5]
vx = vi['x'].astype(float)
vy = vi['y'].astype(float)
vt = vi['t'].astype(float)
vps = np.where(vi['p'])
vt = vi[vps]['t'].astype(float)
v1 = np.where(vt<100000)[0][-1]
v2 = np.where(vt<200000)[0][-1]
v3 = np.where(vt<300000)[0][-1]
for i,j in ([0,v1],[v1,v2],[v2,v3],[v3,vt.shape[0]]):
    vxi = vx[i:j]
    vyi = vy[i:j]
    vti = vt[i:j]
    fig, ax = plt.subplots(1, 1)
    colorplot(vxi,vyi,vti,fig,ax)

