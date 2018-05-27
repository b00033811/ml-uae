import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.animation import FuncAnimation
from Poly import Poly                         
#%% Import Data
x_raw,y_raw=np.loadtxt('cross_data.csv',delimiter=',')

#%%
# Generate polynomial features
degree=12 # k=15
# call the polynomial class and fit
p0=Poly([1]*(degree+1),list(range(0,degree+1)))
p0.fit(x_raw)
# Get the polynomial features
poly_features=p0.poly_features()
# reshape the data
x=np.array(poly_features)
y=y_raw.reshape(-1,1)
# split the model into test/train sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5,
                                               random_state=2,
                                               shuffle=True)
# create models
models=[LinearRegression().fit(x_train[:,0:i+1],y_train)\
        for i in range(0,degree)]


#%% animation plot
n_points=200
p_plot=Poly([1]*(degree+1),list(range(0,degree+1)))
p_plot.fit(np.linspace(0,max(x_raw),n_points))
x_=np.array(p_plot.poly_features())
predictions=[models[i].predict(x_[:,0:i+1])\
             for i in range(0,degree)]
fig,ax=plt.subplots()
lines, =ax.plot([],[])
ax.scatter(x_test[:,1],y_test,color='g',label='Testing data')
ax.scatter(x_train[:,1],y_train,color='b',label='Training data')
plt.legend(loc=2)
ax.grid()
ax.set_aspect(.5)

def update(frame):
    ax.set_title('Polynomial Degree = {}'.format(frame))
    lines.set_data(x_[:,1],predictions[frame])
    return lines,
def init():
    ax.set_xlim(-1,max(x_raw)+.5)
    ax.set_ylim(-1,max(y_raw)+.5)
    return lines,

anim=FuncAnimation(fig,update,frames=range(0,degree),interval=500,\
                   init_func=init,
                   blit=False,
                   repeat=False)

anim.save('overfitting.gif',dpi=120,writer='imagemagick')
plt.show()



