# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:26:00 2018

@author: b0003
"""
import numpy as np
np.random.seed(1337)
from NaiveBayes_log import NB
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
def data_gen(param,n,shuffle='False'):
    data=np.empty((0,2))
    for u,v,l in param:
        x=np.random.normal(u,v,n).reshape(n,1)
        y=np.full((n,1),l)
        data=np.append(data,np.concatenate([x,y],axis=1),axis=0)
    if shuffle=='True':
        np.random.shuffle(data)
        return data
    return data


param=[[-2,1,0],[2,1,1]]
raw_data=data_gen(param,1000,shuffle='True')
x=raw_data[:,0]
r=raw_data[:,1].reshape(-1,1)
encoder=OneHotEncoder()
r=encoder.fit_transform(r).toarray()
model=NB(x,r)
H=model.fit()
x=np.linspace(-6,6,1000)
p=model.posterior(x)
l=model.likelihood(x)
fig,ax=plt.subplots(2,1)
for axs in ax:
    axs.grid()
ax[0].plot(x,p)
ax[0].set_xlabel('x')
ax[0].set_ylabel('Posterior')
ax[1].plot(x,np.exp(l))
ax[1].set_xlabel('x')
ax[1].set_ylabel('Liklelihood')
plt.show()
plt.tight_layout()
