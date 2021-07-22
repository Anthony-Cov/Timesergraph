"""
Creating a table with mean MAPE for different models, Graph characteristics
and time-series features.
"""
import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from time import time
from Libraries.Util import seconds_to_str, RemTrend, Norm01
from Libraries.Autoregr import VARExplore
from Libraries.Spectrum import MSSAExplore
from Libraries.Localapp import LAprExplore
from Libraries.ChooChoo import ChooChooExplore
from Libraries.NeurosV import LSTMExploreV
from Libraries.graph import MakeGraph, GraphEntropy
from Libraries.features import NoiseFactor, CНurst, KSEntr, RandWalk, ShEntr

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

datadir='RealWeekly' #name for data folder
filename='real_table.csv' #name roe results
#datadir='Art_series' #name for data folder
#filename='art_table.csv' #name roe results
datafiles=os.listdir(datadir)
try: # If no table found we'll craeate it otherwise we'll continue
    table=pd.read_csv(filename)
except:
    print('No table, let\'s create it')
    table=pd.DataFrame(columns=['Series', 'VARmape', 'LAmape', 'MSSAmape', 'Choomape', 'RNNmape',
                                'Gsize', 'Grent', 'Connect', 'Assort','Density', 'Modularity', 'Cycles', 
                                'Noise', 'Hurst', 'KSent','Rndwl', 'Corent'])
begin=len(table)
print('Table contains %d items'%begin)
t=time()
for i,df in enumerate(datafiles[begin:]): #it may take much time - RNN is not so fast
    '''Reading data file '''
    print(i, df[:-4], '\tTime:', seconds_to_str(time()-t))
    data=pd.read_csv(datadir+'/'+df, sep=',')
    data[data.columns[1]]=pd.to_numeric(data[data.columns[1]], errors='coerce', downcast='float')
    x=data[data.columns[1]].values
    x=np.delete(x, np.where(x=='.')).astype(float)
    l=len(x)
    
    '''Time series forecasting wuth various methods '''
    mapes1,mapes2, mapes3, mapes4, mapes5 =[],[],[],[],[]
    for split in range(12, 132, 6): #forecasting tests on different parts of a series
        m=VARExplore(x,[],12,split)[2]
        if m<300: 
            mapes1.append(m)
        print('a', end='')
        m=LAprExplore(x,[],12,split)[2]
        if m<300: 
            mapes2.append(m)
        print('l', end='')
        m=MSSAExplore(x,[],12,split)[2]
        if m<300: 
            mapes3.append(m)
        print('s', end='')
        m=ChooChooExplore(x,[],12,split)[2]
        if m<300: 
            mapes4.append(m)
        print('c', end='')
        m=LSTMExploreV(pd.Series(x),[],12,split)[2] #the slowest methon, can be excluded
        if m<300: 
            mapes5.append(m)
        print('n|', end='')
    print('*')
    
    '''Graph and its characteristics'''
    z, a,b = RemTrend(x[:])
    z=Norm01(z)[0]
    G=MakeGraph(z, mindist=.01, method='hierarchy', k=25)#'KMeans'Norm01(z)[0]
    gsize=G.number_of_edges()/G.number_of_nodes()
    grent=GraphEntropy(G)
    gconnect=nx.average_node_connectivity(G)
    gassort=nx.degree_assortativity_coefficient(G)
    gdens=nx.density(G)
    com=[i for i in list(nx.algorithms.community.modularity_max._naive_greedy_modularity_communities(G)) if len(i)>1]
    gmodul=len(com)/G.number_of_nodes()
    c1=len(list(i for i in nx.cycles.simple_cycles(G) if (len(i) > G.number_of_nodes()//5)))
    c0=len(list(nx.cycles.simple_cycles(G)))
    gcycle=c1/c0
    
    '''Time series features'''
    z=Norm01(x)[0]
    noise=NoiseFactor(z)
    hurst=CНurst(z)
    ksent=KSEntr(z)
    rndw=RandWalk(z)
    corrent=ShEntr(z)
    
    '''New row in the table'''
    table=table.append(pd.DataFrame({'Series':[df[:-4]], 'VARmape':[np.mean(mapes1)], 
                                     'LAmape':[np.mean(mapes2)], 'MSSAmape':[np.mean(mapes3)],
                                     'Choomape':[np.mean(mapes4)], 'RNNmape':[np.mean(mapes5)],
                                     'Gsize':[gsize], 'Grent':[grent],	
                                     'Connect':[gconnect], 'Assort':[gassort],
                                     'Density':[gdens], 'Modularity':[gmodul],
                                     'Cycles':[gcycle], 'Noise':[noise], 
                                     'Hurst':[hurst], 'KSent':[ksent],
                                     'Rndwl':[rndw], 'Corent':[corrent]}), ignore_index=True)
    '''Save every single step  '''
    table.to_csv(filename, index=False)
print('Done! Time:', seconds_to_str(time()-t))