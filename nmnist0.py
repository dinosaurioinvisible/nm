
import os
import numpy as np
import matplotlib.pyplot as plt 
from eventvision import Events, read_dataset
from neuro_dataset import generate_nmnist_dataset



def mk_dataset(num=5, nfiles=100, dst='Train'):
    dirpath = os.path.abspath(os.path.join(os.getcwd(),'..','nmnist_data',dst,str(num)))
    fnames = [fn for fn in os.listdir(dirpath)]
    ds = np.rec.array(None, dtype=[('imdata',np.ndarray),
                               ('num',np.uint32),
                               ('evs',np.uint16),
                               ('height',np.uint16),
                               ('width',np.uint16)],
                               shape=(nfiles))
    for ei,fname in enumerate(fnames[:nfiles]):
        filepath = os.path.join(dirpath,fname)
        td = read_dataset(filename=filepath)
        ds[ei].imdata = td.data
        ds[ei].num = num
        ds[ei].evs = td.data.shape[0]
        ds[ei].height = td.height
        ds[ei].width = td.width
    return ds
ds = mk_dataset(num=5, nfiles=100)

def mk_pn_mws(dsx,ms=5000): # just run the fx twice
    pts = dsx.imdata.ts[np.where(dsx.imdata.p)]
    nts = dsx.imdata.ts[np.where(dsx.imdata.p==False)]
    dpt = pts[1:] - pts[:-1]
    dnt = nts[1:] - nts[:-1]
    cp,cn = 0,0
    pis,nis = [],[]
    for i,(px,nx) in enumerate(zip(dpt,dnt)): # one is longer than the other
        cp = 0 if cp > 5000 else cp + px
        cn = 0 if cn > 5000 else cn + nx
        print(i,len(pts),cp,px, i,len(nts),cn,nx)
mk_pn_mws(ds[0])