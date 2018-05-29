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
# Get the polynomial features
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
kf=KFold(4)
# Initiate classifiers 
models=[LinearRegression() for m in range(0,degree)]
scores=[] #variable to store the final scores
for k,m in enumerate(models):
    v_temp=[] # temp. variable, stores validation fold(s) result.
    t_temp=[] # temp. variable, stores training fold(s) result.
    # get indx. for valid and training
    # NOTE: [:,0:k+1] fit only on features corrosponding to order.
    for i_valid,i_train in kf.split(_x): 
        m.fit(_x[i_train][:,0:k+1],\
              _y[i_train])
        # get score for current fold and store it
        t_temp.append(m.score(_x[i_train][:,0:k+1],_y[i_train]))
        v_temp.append(m.score(_x[i_valid][:,0:k+1],_y[i_valid]))
    scores.append([t_temp,v_temp])
# get the avg. error for testing and valid folds.
test_error=[1-sum(t)/len(t) for t,_ in scores]
valid_error=[1-sum(v)/len(v) for _,v in scores]
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
y_pred=models[k].predict(x[:,0:k+1])
ax.plot(x_raw,y_pred,color='b',\
         label='Polynomial model k={1} | R2={0:.2f}'.format(k_score,k))
ax.scatter(x_raw,y_raw,color='r',marker='.',label='Input data')
ax.grid(),ax.legend()
fig2,ax2=plt.subplots()
ax2.plot(test_error,label='Test Error')
ax2.plot(valid_error,label='Valid Error')
ax2.grid(),ax2.legend()

