"""
Polynomials
@author: Abdullah Alnauimi
"""
import numpy as np
import matplotlib.pyplot as plt

def fit_poly(a,k):
    '''returns a function (A=V.a) for every term in a polynomial\
    where V is the vandermonde matrix, a is the coef. matrix'''
    A=[lambda x,a=a,k=k:[a*n**k for n in x] for a,k in zip(a,k)]
    return A
def evaluate_poly(x,functions):
    '''# evaluates A=V.a,stores it in matrix form, and returns a list y(x)=[A0,..Ak]'''
    linear_combinations=[A(x) for A in functions]
    return [sum(i) for i in zip(*linear_combinations)] ,\
                             list(map(list,zip(*linear_combinations)))
############################## Main #########################################
# y(x)=1+x+x^2
coefficients=[0,1,1,-.2] # polynomial 
degree=[0,1,2,3]
A=fit_poly(coefficients,degree) # returns functions A0 A1 A2
# Evaluate the functions and returns y(x)
x=np.linspace(0,5,100)
p,_=evaluate_poly(x,A) 
# Fix the random seed and add gauss. noise to the data.
np.random.seed(seed=1337)
e=np.random.normal(0,.7,len(x))
y=p+e
#Plotting
plt.scatter(x,y,label='y(x)=x+x^2-0.2x^3+e(x)',marker='.')
plt.plot(x,p,label='p(x)=x+x^2-0.2x^3',color='red')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.grid()
np.savetxt('data.csv', (x,y), delimiter=',')