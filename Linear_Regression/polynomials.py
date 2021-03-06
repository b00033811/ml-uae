"""
Polynomials
@author: Abdullah Alnuaimi
"""
import numpy as np
import matplotlib.pyplot as plt
def fit_poly(a,k):
    '''returns a function of the dot product (A=V.a) '''
    A=lambda x,a=a,k=k:[[a*n**k for a,k in zip(a,k)] for n in x]
    return A
def evaluate_poly(x,A):
    ''' evaluates A=V.a,stores it in matrix form, and 
     returns a list y(x)=[A0,..An]'''
    y=[sum(i) for i in A(x)] 
    return y,A(x)
############################## Main #########################################
# y(x)=x+x^2-0.2x^3
coefficients=[0,1,1,-.2] # polynomial 
degree=[0,1,2,3]
A=fit_poly(coefficients,degree) # returns A(x0)...A(Xn) 
# Evaluate the functions and returns y(x)
x=np.linspace(0,5,20)
y,_=evaluate_poly(x,A) 
#Plotting
plt.plot(x,y,label='p(x)=x+x^2-0.2x^3')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.grid()