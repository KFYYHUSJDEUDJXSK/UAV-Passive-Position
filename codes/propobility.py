import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
config = {"font.family":'Times New Roman',"font.size": 16,"mathtext.fontset":'stix'}
rcParams.update(config)

import pandas as pd
_dr_actual = [np.random.uniform(0,0.16),np.random.uniform(0, 2*np.pi)]
rou = np.random.uniform(0,1,size=(10000,))
rou = rou **0.5 * 0.16
theta = np.random.uniform(0, 2*np.pi,size=(10000,))
x = np.array([])
y = np.array([])
# for _rou,_theta in zip(rou, theta):
#     x = np.append(x,_rou * np.cos(_theta))
#     y = np.append(y,_rou * np.sin(_theta))
x = np.array([_rou * np.cos(_theta) for _rou,_theta in zip(rou, theta)])
y = np.array([_rou * np.sin(_theta) for _rou,_theta in zip(rou, theta)])
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
fig,ax=plt.subplots(figsize=(12,9),dpi=100)
scatter=ax.scatter(x,y,marker='o',c=z,s=15,label='LST',cmap='Spectral_r')
cbar=plt.colorbar(scatter,shrink=1,orientation='vertical',extend='both',pad=0.015,aspect=30,label='frequency') #orientation='horizontal'
font3={'family':'SimHei','size':16,'color':'k'}
plt.ylabel("估计值",fontdict=font3)
plt.xlabel("预测值",fontdict=font3)
# plt.savefig('F:/Rpython/lp37/plot70.png',dpi=800,bbox_inches='tight',pad_inches=0)
plt.show()