'''Artificial time-series generator'''
import numpy as np
import pandas as pd
from os import path, makedirs, listdir, remove
from itertools import product

def trans(n):
    x=np.linspace(-n*.034, n*.01, n)
    y=1-1/(1+np.exp(x))
    return y
def period(n):
    x=np.linspace(0,16*np.pi, n)
    y=np.sin(x)/2+.5
    return y
def noise(n):
    y=np.random.randn(n)
    y=(y-min(y))/(max(y)-min(y))
    return y
def rndwalk(n):
    y=[0.]
    for i in range(n-1):
        k=np.random.rand()
        sign = 1. if np.random.randint(2) else -1.
        y.append(y[-1]+sign*k)
    y=np.array(y)
    y=(y-min(y))/(max(y)-min(y))
    return y
def compose(n,kt,kp,kn,kr):
    y=kt*trans(n) + kp * period(n) + kn * noise(n) + kr * rndwalk(n)
    y=(y-min(y))/(max(y)-min(y))
    return y
    
artdir='Art_series' # new folder for the artificial set
n=750 #length of a series
if not path.exists(artdir):
    makedirs(artdir)
else:
    for f in listdir(artdir):
        remove(artdir+'/'+f)

''' По всем параметрам :) '''
i=0
for k in product([0.0, .2, .5, 1.], repeat=4): 
    if sum(k) < 0.5:
        continue
    y=compose(n,k[0],k[1],k[2],k[3])
    y=(y-min(y))/(max(y)-min(y))
    ser=pd.DataFrame({'t':np.arange(n), 'val':y})
    ser.to_csv(artdir+'/'+str(i).zfill(4)+'.csv', index=False)
    i+=1