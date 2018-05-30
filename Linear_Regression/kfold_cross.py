import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Poly import Poly   
from sklearn.model_selection import KFold                      
#%% Import Data
x_raw,y_raw=np.loadtxt('data.csv',delimiter=',')
# k = 8
degree=8 # k=8
# Initialize the polynomial class and fit
p0=Poly([1]*(degree+1),list(range(0,degree+1)))
p0.fit(x_raw)
# Get the polynomial features for 8th order model
poly_features=p0.poly_features()
# reshape the data
x=np.array(poly_features)
y=y_raw.reshape(-1,1)
# split the model into test/train sets
_x,x_test,_y,y_test=train_test_split(x,y,test_size=.3,
                                               random_state=1337,
                                               shuffle=True)
#%% Split into Kfold and get the test/valid score.
# Create 4 folds
folds=4
kf=KFold(folds)
# Initiate classifiers 
models=[LinearRegression() for m in range(0,degree)]
valid_error=[] # variable to store the valid error
train_error=[] # variable to store the train error
#looping on every model.
for k,m in enumerate(models):
    # get poly features for current order [k] poly. model
    k_features=_x[:,0:k+1]
    # function: fit and get r2 score of model.
    f=lambda t,v:m.fit(k_features[t],_y[t]).\
             score(k_features[v],_y[v])
    # validation scores for each fold
    v_score=[f(train,valid) for valid,train in kf.split(_x)]
    # training score for each fold
    t_score=[f(train,train) for _,train in kf.split(_x)]
    # average valid and test error for all folds
    valid_error.append(1-sum(v_score)/len(v_score))
    train_error.append(1-sum(t_score)/len(t_score))
# get best model poly. order
k=np.argmin(valid_error)
# fit model to training and validation data
models[k].fit(_x[:,0:k+1],_y)
# score model against testing data
k_score=models[k].score(x_test[:,0:k+1],y_test)
print('Optimum polynomial order = {0}\n\
testing score = {1:.2f}'.format(k,k_score))
#%% plotting
fig1,ax=plt.subplots()
ax.set_xlabel('x'),ax.set_ylabel('f(x)')
y_pred=models[k].predict(x[:,0:k+1])
ax.plot(x_raw,y_pred,color='b',\
         label='Polynomial model k={1} | R2={0:.2f}'.format(k_score,k))
ax.scatter(x_raw,y_raw,color='r',marker='.',label='Input data')
ax.grid(),ax.legend()
fig2,ax2=plt.subplots()
ax2.set_xlabel('Polynomial Order'),ax2.set_ylabel('Error')
ax2.plot(train_error,label='Train Error')
ax2.plot(valid_error,label='Valid Error')
ax2.grid(),ax2.legend()