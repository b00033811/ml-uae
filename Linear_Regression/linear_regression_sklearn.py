import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# loading data
x_raw,y_raw=np.loadtxt('data.csv',delimiter=',')
# reshape the data
n_features=1
x=x_raw.reshape(-1,n_features)
y=y_raw.reshape(-1,n_features)
# split the model into test/train sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,
                                               random_state=1337,
                                               shuffle=True)
# fit the linear model
model=LinearRegression()
model.fit(x_train,y_train)
# get the R2 score
R2=model.score(x_test,y_test)
print ('The R2 score is {0:.2f}'.format (R2))
#get model parameters
coef=model.intercept_,*model.coef_
print (', '.join('a{0}={1:.2f}'.format(i,*a) for i,a in enumerate(coef)))
# get the model predictions &
# visualize the model
predictions=model.predict(x)
fig1=plt.figure('Linear Regression')
plt.plot(x,predictions,color='b',label='Linear model | R2={0:.2f}'.format(R2))
plt.scatter(x_raw,y_raw,color='r',marker='.',label='Input data')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()
plt.title('Linear Regression Using sklearn')
