# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:37:01 2019
Тесты Грейнджера, кросс-корреляции, CCM и заодно VAR 
@author: pop.antonij@gmail.com
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from babavanga.Autoregr import VARModel
from babavanga.Util import Norm01
from babavanga.Util import Nback
from babavanga.Util import Metr
from babavanga.Util import MovingAverage
'''Тест Грейнджера'''
def GrangerTest(x1,x2):
    x=pd.DataFrame()
    x['res'],_,_=Norm01(x1)
    x['cause'],_,_ =Norm01(x2)
    gr_test=sm.tsa.stattools.grangercausalitytests(x, maxlag=12, verbose=False)
    p1=min([gr_test[j][0]['ssr_ftest'][1] for j in gr_test])
    return(p1)
'''Выбор n лучших по Грейнджеру'''    
def ChoosePredsGran(region, predictors, n):
    goodpr=[]
    for j,i in enumerate(predictors):
        predictor=pd.read_csv('Data_predictors-II/'+i, sep=',')
#    p1.index(min(p1))+1, min(p1)
        x=pd.DataFrame()
        x['res']=region-MovingAverage(region)
        x['res'],_,_=Norm01(x['res'])
        x['cause'] = predictor[predictor.columns[1]].values-MovingAverage(predictor[predictor.columns[1]])
        x['cause'],_,_ =Norm01(x['cause'])
        gr_test=sm.tsa.stattools.grangercausalitytests(x, maxlag=12, verbose=False)
        p1=[gr_test[j][0]['ssr_ftest'][1] for j in gr_test]
        goodpr.append({'prd':i, 'score': min(p1), 'lag': p1.index(min(p1))+1})
    return sorted(goodpr, key = lambda i: i['score'])[:n]

'''Кросс-корреляция'''
def CrossCorr(datax, datay, maxlag=12):
    ccor=0
    lag=0
    dy=pd.Series(datay)
    for i in range(1,maxlag):
        c=abs(datax.corr(dy.shift(i),method='spearman'))
        if c>ccor:
            ccor=c
            lag=i
    return lag,ccor 
'''Выбор n лучших по кросс-корреляции''' 
def ChoosePredsCCor(region, predictors, n):
    pred=[]
    for iprd in predictors:
        predictor=pd.read_csv('Data_predictors-II/'+iprd, sep=',')
        scf=max(region)
        x1=region/scf
        x2,_,_=Norm01(predictor[predictor.columns[1]])
        lag, ccor=CrossCorr(x1, x2, maxlag=12)
        pred.append({'prd':iprd,'lag':lag,'score':ccor})
    return sorted(pred, key = lambda i: i['score'], reverse=True)[:n]

'''Хорошие по CCM'''
import skccm as ccm
from skccm.utilities import train_test_split
import warnings

def CCMTest(x1,x2):
    warnings.filterwarnings("ignore")
#выбрать lag
    lag,_=CrossCorr(pd.Series(x1),pd.Series(x2) , maxlag=11)
    CCM = ccm.CCM(score_metric='corrcoef') #вариант 'score'
    embed = 2
    scr=0
#выбрать embed
    for i in range(2,5):
        e1 = ccm.Embed(x1)
        e2 = ccm.Embed(x2)
        X1 = e1.embed_vectors_1d(lag,i)
        X2 = e2.embed_vectors_1d(lag,i)
        x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.9)
        len_tr = len(x1tr)
        lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')
        CCM.fit(x1tr,x2tr)
        x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)
        sc1,sc2 = CCM.score()
        if max(sc1)>scr:
            embed, scr = i, max(sc1) 
    return(lag, embed, scr)
'''Выбор n лучших по CCM''' 
def ChoosePredsCCM(region, predictors, n):
    pred=[]
    for iprd in predictors:
        predictor=pd.read_csv('Data_predictors-II/'+iprd, sep=',')
        scf=max(region)
        x1=region/scf
        #x2=predictor[predictor.columns[1]]
        #x1,_,_=Norm01(data[data.columns[idat]])
        x2,_,_=Norm01(predictor[predictor.columns[1]])
        lag, embed, score=CCMTest(x1,x2)
        pred.append({'prd':iprd,'lag':lag,'embed':embed, 'score':score})
    return sorted(pred, key = lambda i: i['score'], reverse=True)[:n]
'''Хорошие по VAR'''
def VARTest(x,y,maxlag=12):
    b=len(x)-6
    fwd=6
    score=200
    lag=0    
    for l in range(maxlag):
        x1,mi,ma=Norm01(x)
        x1=x1[l:]
        y1,_,_=Norm01(y)
        y1=pd.Series(y1).shift(l)[l:]
        vec=pd.DataFrame({'reg':x1, 'prd':y1})
        x_test=Nback(VARModel(vec[:b],fwd), mi,ma)
        d = Metr(x[b:b+fwd], x_test)
        if d[2]<score:
            score=d[2]
            lag=l
    return score, lag
'''Выбор n лучших по VAR''' 
def ChoosePredsVAR(dat, predictors, n):
    pred=[]
    for j in predictors:
        predictor=pd.read_csv('Data_predictors-II/'+j, sep=',')
        prd=predictor[predictor.columns[1]]
        s,l=VARTest(dat,prd,maxlag=12)
        pred.append({'prd':j,'lag':l,'score':s})
    pred=sorted(pred, key = lambda i: i['score'], reverse=False)
    return pred[:n]