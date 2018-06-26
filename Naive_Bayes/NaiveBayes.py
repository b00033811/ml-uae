"""
Univariate Naive Bayes

@author: Abdullah Al Nuaimi
"""
import numpy as np
class NB():
    def __init__(self,x,r):
        # define the input data
        self.x=x
        self.r=r
        # define the number of data points (t) and hypotheses (i)
        self.t,self.i=x.shape[0],r.shape[1]
        # initialize a hypotheses set with 2 parameters mean and var
        self.H=np.empty((self.i,3))
    def fit(self):
        for i in range(0,self.i):
            mean=np.average(self.x[self.r[:,i]==1])
            var=np.var(self.x[self.r[:,i]==1],ddof=True)
            prior=sum(self.r[:,i]==1)/len(self.r[:,i])
            self.H[i,:]=np.array([mean,var,prior])
        return self.H
    def likleyhood(self,x):
        ''' calculate the likelyhood of data over all H'''
        L=np.empty((len(x),self.i))
        for idx,h in enumerate(self.H):
            u=h[0]
            v=h[1]
            l=(-1/2)*np.log(2*np.pi)-np.log(v)-((x-u)**2/(2*v**2))
            L[:,idx]=l
        return np.exp(L)
    def evidence(self,x):
        return np.sum(np.dot(self.likleyhood(x),self.H[:,2].reshape(self.i,-1)),axis=1)
    def posterior(self,x):
        return (self.likleyhood(x)*self.H[:,2])/self.evidence(x)[:,None]
    def predict(self,x):
        g= -(-self.H[:,0]+x[:,None])**2
        return np.eye(self.i)[np.argmax(g,axis=1)]

        
        
        
        
        
        