# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:24:27 2021
Метод Чучуевой
dat - исходный временной ряд  
prds - предикторы /пока не актуальны, пиши []
fwd - горизонт прогнозирования
a0 - True/False - искать ли свободный член;
Возвращает продолжение ряда до означенного горизонта
@author: user
"""

import numpy as np
import pandas as pd
from Libraries.Util import Metr
from Libraries.features import CEmbDim
def ChooChoo(dat, prds, fwd, a0=True):
    m=2*CEmbDim(dat)+1
    z0=dat[-m:]
    dz0=np.std(z0)
    l, k = 0, 0
    for i in range(len(dat)-(m+fwd+1),0,-1): #с какого конца начать ?2*m-weeklen
        z=dat[i:i+m]
        cor=np.cov(z0, z)[0,1]/np.std(z)/dz0
        if cor>l:
            l,k=cor,i
    zk=dat[k:k+m].reshape(m,1)
    xhat=[]
    for i,j in enumerate(prds):
        predictor=pd.read_csv('Data_predictors-II/'+j['prd'], sep=',')
        x=predictor[predictor.columns[1]].values[:-j['lag']]
        zk=np.concatenate((zk,x[-m:].reshape(m,1)),axis=1)
        m1=2*CEmbDim(x)+1
        xhat.append(ChooChoo(x, [], m1, fwd, a0=a0))
    if a0: 
        zk=np.concatenate((zk,np.ones((m,1))),axis=1)
    a=(zk.T.dot(z0)).dot(np.linalg.inv(zk.T.dot(zk)))
    if not a0: a=np.concatenate((a, np.zeros(1)))
    zhat=dat[k+m:k+m+fwd]*a[0]+a[-1]
    for i in range(len(prds)):
        zhat+=a[i+1]*xhat[i]
    return zhat
def ChooChooExplore(dat,prds,fwd,split):
    b=len(dat)-split
    z=ChooChoo(dat[:b],prds,fwd)
    m,d1np,d2np,d3np,d4 = Metr(z, dat[b:b+fwd])
    return m,d1np,d2np,d3np,d4,z
def ChooChooUse(dat,prds,fwd):
    return ChooChoo(dat, prds, fwd)
