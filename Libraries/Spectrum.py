﻿# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:40:00 2019
Прогнозирование временного ряда MSSA
@author: pop.antonij@gmail.com
"""
import numpy as np
import pandas as pd
import warnings
from pymssa import MSSA
from Libraries.Util import Norm01
from Libraries.Util import Nback
from Libraries.Util import Metr

def MSSAModel(x, fwd):
    warnings.filterwarnings("ignore")
    mssa = MSSA(n_components=7, #'variance_threshold',
            variance_explained_threshold=0.98,#,'svht',
            window_size=None,
            verbose=False)
    xn,mi,ma=Norm01(x.iloc[:,0])
    x.iloc[:,0]=xn 
    mssa.fit(x)
    tr = np.zeros(x.shape[0])
    tr[:] = np.nan
    tr = mssa.components_[0, :, :].sum(axis=1)
    fc = mssa.forecast(fwd, timeseries_indices=0)
    tr=Nback(tr, mi,ma)
    fc=Nback(fc, mi,ma)
    return tr, fc, 

def MSSAExplore(dat,prds,fwd,split):
    x=GetDataPred(dat,prds)
    b=len(x)-split
    x1=x[:b]   
    tr, fc=MSSAModel(x1, fwd)
    m,d1np,d2np,d3np,d4=Metr(fc[0], x.iloc[b:b+fwd, 0])
    y=np.concatenate((tr, fc[0]), axis=None)
    return m,d1np,d2np,d3np,d4,y 

def MSSAUse(dat,prds,fwd):
    x=GetDataPred(dat,prds)
    _,fc=MSSAModel(x,fwd)
    return fc[0]

def GetDataPred(d,prds):
    x=pd.DataFrame()
    if len(prds):
        maxlag=max([p['lag'] for p in prds])
        x['reg']=d[maxlag:]
        for i,j in enumerate(prds):           
            predictor=pd.read_csv('Data_predictors-II/'+j['prd'], sep=',')
            xx=predictor[predictor.columns[1]].shift(j['lag'])[maxlag:]
            x['prd'+str(i)]=Norm01(xx.values)[0]
    else:
        x['reg']=d
    return x
