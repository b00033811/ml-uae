import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
                             
# loading data
x_raw,y_raw=np.loadtxt('data.csv',delimiter=',')
# generate polynomial features
degree=3
poly_features=fit_poly([1]*(degree+1),list(range(0,degree+1)))
_,poly_features=evaluate_poly(x_raw,poly_features)
x=np.array(poly_features)
# reshape the data
y=y_raw.reshape(-1,1)
# split the model into test/train sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,
                                               random_state=1337,
                                               shuffle=True)
# fit the linear model
model=LinearRegression()
model.fit(x_train,y_train)
# get the R2 score
R2=model.score(x_test,y_test)
print ('The R2 score is {0:.3f}'.format (R2))
#get model parameters
coef=model.coef_.T
print (', '.join('a{0}={1:.2f}'.format(i,*a) for i,a in enumerate(coef)))
# get the model predictions &
# visualize the model
predictions=model.predict(x)
fig1=plt.figure('Linear Regression')
plt.plot(x_raw,predictions,color='b',label='Polynomial model | R2={0:.2f}'.format(R2))
plt.scatter(x_raw,y_raw,color='r',marker='.',label='Input data')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()
plt.title('Polynomial Regression Using sklearn')

