#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


# In[10]:




EVGO_Mnist_lossepoch = np.loadtxt(open("EVGO_Mnist_lossepoch.csv", "rb"),  delimiter="," )
EVGO_Mnist_loss_tra = np.loadtxt(open("EVGO_Mnist_loss_tra.csv", "rb"),  delimiter="," )
EVGO_Mnist_acc_tra = np.loadtxt(open("EVGO_Mnist_acc_tra.csv", "rb"),  delimiter="," )
EVGO_Mnist_loss_test = np.loadtxt(open("EVGO_Mnist_loss_test.csv", "rb"),  delimiter="," )
EVGO_Mnist_acc_test = np.loadtxt(open("EVGO_Mnist_acc_test.csv", "rb"),  delimiter="," )
EVGO_Mnist_loss_val = np.loadtxt(open("EVGO_Mnist_loss_val.csv", "rb"),  delimiter="," )
EVGO_Mnist_acc_val = np.loadtxt(open("EVGO_Mnist_acc_val.csv", "rb"),  delimiter="," )

#np.loadtxt("xCR2_EVGO3_Mnist_lossepoch_test.csv", "rb") lossepoch_test, delimiter="," )
#np.loadtxt("xCR2_EVGO3_Mnist_skip.csv", "rb") skip, delimiter="," )
EVGO_MNIST_final_loss = np.loadtxt(open("EVGO_MNIST_final_loss.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_acc = np.loadtxt(open("EVGO_MNIST_final_acc.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_loss_val = np.loadtxt(open("EVGO_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_acc_val = np.loadtxt(open("EVGO_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_loss_test = np.loadtxt(open("EVGO_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_acc_test = np.loadtxt(open("EVGO_MNIST_final_acc_test.csv", "rb"),  delimiter="," )

EVGO_MNIST_final_loss_2d = np.loadtxt(open("2d_EVGO_MNIST_final_loss.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_acc_2d = np.loadtxt(open("2d_EVGO_MNIST_final_acc.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_loss_val_2d=np.loadtxt(open("2d_EVGO_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_acc_val_2d =np.loadtxt(open("2d_EVGO_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_loss_test_2d=np.loadtxt(open("2d_EVGO_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
EVGO_MNIST_final_acc_test_2d = np.loadtxt(open("2d_EVGO_MNIST_final_acc_test.csv", "rb"),  delimiter="," )


# In[11]:




GD_Mnist_lossepoch = np.loadtxt(open("GD_Mnist_lossepoch.csv", "rb"),  delimiter="," )
GD_Mnist_loss_tra = np.loadtxt(open("GD_Mnist_loss_tra.csv", "rb"),  delimiter="," )
GD_Mnist_acc_tra = np.loadtxt(open("GD_Mnist_acc_tra.csv", "rb"),  delimiter="," )
GD_Mnist_loss_test = np.loadtxt(open("GD_Mnist_loss_test.csv", "rb"),  delimiter="," )
GD_Mnist_acc_test = np.loadtxt(open("GD_Mnist_acc_test.csv", "rb"),  delimiter="," )
GD_Mnist_loss_val = np.loadtxt(open("GD_Mnist_loss_val.csv", "rb"),  delimiter="," )
GD_Mnist_acc_val = np.loadtxt(open("GD_Mnist_acc_val.csv", "rb"),  delimiter="," )

#np.loadtxt("xCR2_GD3_Mnist_lossepoch_test.csv", "rb") lossepoch_test, delimiter="," )
#np.loadtxt("xCR2_GD3_Mnist_skip.csv", "rb") skip, delimiter="," )
GD_MNIST_final_loss = np.loadtxt(open("GD_MNIST_final_loss.csv", "rb"),  delimiter="," )
GD_MNIST_final_acc = np.loadtxt(open("GD_MNIST_final_acc.csv", "rb"),  delimiter="," )
GD_MNIST_final_loss_val = np.loadtxt(open("GD_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
GD_MNIST_final_acc_val = np.loadtxt(open("GD_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
GD_MNIST_final_loss_test = np.loadtxt(open("GD_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
GD_MNIST_final_acc_test = np.loadtxt(open("GD_MNIST_final_acc_test.csv", "rb"),  delimiter="," )

GD_MNIST_final_loss_2d = np.loadtxt(open("2d_GD_MNIST_final_loss.csv", "rb"),  delimiter="," )
GD_MNIST_final_acc_2d = np.loadtxt(open("2d_GD_MNIST_final_acc.csv", "rb"),  delimiter="," )
GD_MNIST_final_loss_val_2d=np.loadtxt(open("2d_GD_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
GD_MNIST_final_acc_val_2d =np.loadtxt(open("2d_GD_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
GD_MNIST_final_loss_test_2d=np.loadtxt(open("2d_GD_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
GD_MNIST_final_acc_test_2d = np.loadtxt(open("2d_GD_MNIST_final_acc_test.csv", "rb"),  delimiter="," )


# In[12]:




RmsProp_Mnist_lossepoch = np.loadtxt(open("RmsProp_Mnist_lossepoch.csv", "rb"),  delimiter="," )
RmsProp_Mnist_loss_tra = np.loadtxt(open("RmsProp_Mnist_loss_tra.csv", "rb"),  delimiter="," )
RmsProp_Mnist_acc_tra = np.loadtxt(open("RmsProp_Mnist_acc_tra.csv", "rb"),  delimiter="," )
RmsProp_Mnist_loss_test = np.loadtxt(open("RmsProp_Mnist_loss_test.csv", "rb"),  delimiter="," )
RmsProp_Mnist_acc_test = np.loadtxt(open("RmsProp_Mnist_acc_test.csv", "rb"),  delimiter="," )
RmsProp_Mnist_loss_val = np.loadtxt(open("RmsProp_Mnist_loss_val.csv", "rb"),  delimiter="," )
RmsProp_Mnist_acc_val = np.loadtxt(open("RmsProp_Mnist_acc_val.csv", "rb"),  delimiter="," )

#np.loadtxt("xCR2_RmsProp3_Mnist_lossepoch_test.csv", "rb") lossepoch_test, delimiter="," )
#np.loadtxt("xCR2_RmsProp3_Mnist_skip.csv", "rb") skip, delimiter="," )
RmsProp_MNIST_final_loss = np.loadtxt(open("RmsProp_MNIST_final_loss.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_acc = np.loadtxt(open("RmsProp_MNIST_final_acc.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_loss_val = np.loadtxt(open("RmsProp_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_acc_val = np.loadtxt(open("RmsProp_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_loss_test = np.loadtxt(open("RmsProp_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_acc_test = np.loadtxt(open("RmsProp_MNIST_final_acc_test.csv", "rb"),  delimiter="," )

RmsProp_MNIST_final_loss_2d = np.loadtxt(open("2d_RmsProp_MNIST_final_loss.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_acc_2d = np.loadtxt(open("2d_RmsProp_MNIST_final_acc.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_loss_val_2d=np.loadtxt(open("2d_RmsProp_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_acc_val_2d =np.loadtxt(open("2d_RmsProp_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_loss_test_2d=np.loadtxt(open("2d_RmsProp_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
RmsProp_MNIST_final_acc_test_2d = np.loadtxt(open("2d_RmsProp_MNIST_final_acc_test.csv", "rb"),  delimiter="," )


# In[13]:




Adagrad_Mnist_lossepoch = np.loadtxt(open("Adagrad_Mnist_lossepoch.csv", "rb"),  delimiter="," )
Adagrad_Mnist_loss_tra = np.loadtxt(open("Adagrad_Mnist_loss_tra.csv", "rb"),  delimiter="," )
Adagrad_Mnist_acc_tra = np.loadtxt(open("Adagrad_Mnist_acc_tra.csv", "rb"),  delimiter="," )
Adagrad_Mnist_loss_test = np.loadtxt(open("Adagrad_Mnist_loss_test.csv", "rb"),  delimiter="," )
Adagrad_Mnist_acc_test = np.loadtxt(open("Adagrad_Mnist_acc_test.csv", "rb"),  delimiter="," )
Adagrad_Mnist_loss_val = np.loadtxt(open("Adagrad_Mnist_loss_val.csv", "rb"),  delimiter="," )
Adagrad_Mnist_acc_val = np.loadtxt(open("Adagrad_Mnist_acc_val.csv", "rb"),  delimiter="," )

#np.loadtxt("xCR2_Adagrad3_Mnist_lossepoch_test.csv", "rb") lossepoch_test, delimiter="," )
#np.loadtxt("xCR2_Adagrad3_Mnist_skip.csv", "rb") skip, delimiter="," )
Adagrad_MNIST_final_loss = np.loadtxt(open("Adagrad_MNIST_final_loss.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_acc = np.loadtxt(open("Adagrad_MNIST_final_acc.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_loss_val = np.loadtxt(open("Adagrad_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_acc_val = np.loadtxt(open("Adagrad_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_loss_test = np.loadtxt(open("Adagrad_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_acc_test = np.loadtxt(open("Adagrad_MNIST_final_acc_test.csv", "rb"),  delimiter="," )

Adagrad_MNIST_final_loss_2d = np.loadtxt(open("2d_Adagrad_MNIST_final_loss.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_acc_2d = np.loadtxt(open("2d_Adagrad_MNIST_final_acc.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_loss_val_2d=np.loadtxt(open("2d_Adagrad_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_acc_val_2d =np.loadtxt(open("2d_Adagrad_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_loss_test_2d=np.loadtxt(open("2d_Adagrad_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
Adagrad_MNIST_final_acc_test_2d = np.loadtxt(open("2d_Adagrad_MNIST_final_acc_test.csv", "rb"),  delimiter="," )


# In[14]:




ADAM_Mnist_lossepoch = np.loadtxt(open("ADAM_Mnist_lossepoch.csv", "rb"),  delimiter="," )
ADAM_Mnist_loss_tra = np.loadtxt(open("ADAM_Mnist_loss_tra.csv", "rb"),  delimiter="," )
ADAM_Mnist_acc_tra = np.loadtxt(open("ADAM_Mnist_acc_tra.csv", "rb"),  delimiter="," )
ADAM_Mnist_loss_test = np.loadtxt(open("ADAM_Mnist_loss_test.csv", "rb"),  delimiter="," )
ADAM_Mnist_acc_test = np.loadtxt(open("ADAM_Mnist_acc_test.csv", "rb"),  delimiter="," )
ADAM_Mnist_loss_val = np.loadtxt(open("ADAM_Mnist_loss_val.csv", "rb"),  delimiter="," )
ADAM_Mnist_acc_val = np.loadtxt(open("ADAM_Mnist_acc_val.csv", "rb"),  delimiter="," )

#np.loadtxt("xCR2_ADAM3_Mnist_lossepoch_test.csv", "rb") lossepoch_test, delimiter="," )
#np.loadtxt("xCR2_ADAM3_Mnist_skip.csv", "rb") skip, delimiter="," )
ADAM_MNIST_final_loss = np.loadtxt(open("ADAM_MNIST_final_loss.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_acc = np.loadtxt(open("ADAM_MNIST_final_acc.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_loss_val = np.loadtxt(open("ADAM_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_acc_val = np.loadtxt(open("ADAM_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_loss_test = np.loadtxt(open("ADAM_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_acc_test = np.loadtxt(open("ADAM_MNIST_final_acc_test.csv", "rb"),  delimiter="," )

ADAM_MNIST_final_loss_2d = np.loadtxt(open("2d_ADAM_MNIST_final_loss.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_acc_2d = np.loadtxt(open("2d_ADAM_MNIST_final_acc.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_loss_val_2d=np.loadtxt(open("2d_ADAM_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_acc_val_2d =np.loadtxt(open("2d_ADAM_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_loss_test_2d=np.loadtxt(open("2d_ADAM_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
ADAM_MNIST_final_acc_test_2d = np.loadtxt(open("2d_ADAM_MNIST_final_acc_test.csv", "rb"),  delimiter="," )


# In[19]:




CM_Mnist_lossepoch = np.loadtxt(open("CM_Mnist_lossepoch.csv", "rb"),  delimiter="," )
CM_Mnist_loss_tra = np.loadtxt(open("CM_Mnist_loss_tra.csv", "rb"),  delimiter="," )
CM_Mnist_acc_tra = np.loadtxt(open("CM_Mnist_acc_tra.csv", "rb"),  delimiter="," )
CM_Mnist_loss_test = np.loadtxt(open("CM_Mnist_loss_test.csv", "rb"),  delimiter="," )
CM_Mnist_acc_test = np.loadtxt(open("CM_Mnist_acc_test.csv", "rb"),  delimiter="," )
CM_Mnist_loss_val = np.loadtxt(open("CM_Mnist_loss_val.csv", "rb"),  delimiter="," )
CM_Mnist_acc_val = np.loadtxt(open("CM_Mnist_acc_val.csv", "rb"),  delimiter="," )

#np.loadtxt("xCR2_CM3_Mnist_lossepoch_test.csv", "rb") lossepoch_test, delimiter="," )
#np.loadtxt("xCR2_CM3_Mnist_skip.csv", "rb") skip, delimiter="," )
CM_MNIST_final_loss = np.loadtxt(open("CM_MNIST_final_loss.csv", "rb"),  delimiter="," )
CM_MNIST_final_acc = np.loadtxt(open("CM_MNIST_final_acc.csv", "rb"),  delimiter="," )
CM_MNIST_final_loss_val = np.loadtxt(open("CM_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
CM_MNIST_final_acc_val = np.loadtxt(open("CM_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
CM_MNIST_final_loss_test = np.loadtxt(open("CM_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
CM_MNIST_final_acc_test = np.loadtxt(open("CM_MNIST_final_acc_test.csv", "rb"),  delimiter="," )

CM_MNIST_final_loss_2d = np.loadtxt(open("2d_CM_MNIST_final_loss.csv", "rb"),  delimiter="," )
CM_MNIST_final_acc_2d = np.loadtxt(open("2d_CM_MNIST_final_acc.csv", "rb"),  delimiter="," )
CM_MNIST_final_loss_val_2d=np.loadtxt(open("2d_CM_MNIST_final_loss_val.csv", "rb"),  delimiter="," )
CM_MNIST_final_acc_val_2d =np.loadtxt(open("2d_CM_MNIST_final_acc_val.csv", "rb"),  delimiter="," )
CM_MNIST_final_loss_test_2d=np.loadtxt(open("2d_CM_MNIST_final_loss_test.csv", "rb"),  delimiter="," )
CM_MNIST_final_acc_test_2d = np.loadtxt(open("2d_CM_MNIST_final_acc_test.csv", "rb"),  delimiter="," )


# In[22]:


gd= np.mean(np.array(GD_Mnist_lossepoch).reshape(-1, 100), axis=1)
adam= np.mean(np.array(GD_Mnist_lossepoch).reshape(-1, 100), axis=1)
evgo= np.mean(np.array(GD_Mnist_lossepoch).reshape(-1, 100), axis=1)
cm= np.mean(np.array(GD_Mnist_lossepoch).reshape(-1, 100), axis=1)
rms= np.mean(np.array(GD_Mnist_lossepoch).reshape(-1, 100), axis=1)
adag= np.mean(np.array(GD_Mnist_lossepoch).reshape(-1, 100), axis=1)


alg1.shape


# In[206]:


gd=GD_MNIST_final_loss_2d
adam= ADAM_MNIST_final_loss_2d
evgo=EVGO_MNIST_final_loss_2d
cm= CM_MNIST_final_loss_2d
rms= RmsProp_MNIST_final_loss_2d
adag= Adagrad_MNIST_final_loss_2d

opac=0.6

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

k=9
x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y
zs = np.array(gd)
Z_gd = zs.reshape(X.shape)

zs = np.array(adam)
Z_adam= zs.reshape(X.shape)

zs = np.array(evgo)
Z_evgo = zs.reshape(X.shape)

zs = np.array(cm)
Z_cm = zs.reshape(X.shape)

zs = np.array(rms)
Z_rms = zs.reshape(X.shape)

zs = np.array(adag)
Z_adag = zs.reshape(X.shape)

#ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.plot_surface(X, Y, Z_gd, color='red', alpha=opac, label="GD")
ax.plot_surface(X, Y, Z_adam, color='lightgreen', alpha=opac)
ax.plot_surface(X, Y, Z_cm, color='black', alpha=opac)
ax.plot_surface(X, Y, Z_rms, color='purple', alpha=opac)
ax.plot_surface(X, Y, Z_adag, color='y', alpha=opac)
ax.plot_surface(X, Y, Z_evgo, color='b', alpha=opac)
fig.suptitle("the Loss surfaces for MNIST")


ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('Loss')
ax.margins(x=0, y=0, z=0.9)
#plt.savefig("f_2_fonksiyonu_r3te_y_yegore_turev.png", dpi=600)
ax.view_init(azim = 100, elev = 10)

fake2Dline = mpl.lines.Line2D([0],[0], linestyle="-", c='red')
fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="-", c='lightgreen')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="-", c='y')
fake2Dline3 = mpl.lines.Line2D([0],[0], linestyle="-", c='black')
fake2Dline4 = mpl.lines.Line2D([0],[0], linestyle="-", c='purple')
fake2Dline5 = mpl.lines.Line2D([0],[0], linestyle="-", c='b')
ax.legend([fake2Dline, fake2Dline1, fake2Dline2,fake2Dline3,fake2Dline4,fake2Dline5], ['GD', 'Adam', 'CM', 'RmsProp', 'Adagrad',  'EVGO'], numpoints = 1)
ax.set_zlim(-3,12)

plt.savefig("the Loss surfaces for MNIST.png", dpi=600)
plt.show()


# In[207]:


gd=GD_MNIST_final_loss_2d
adam= ADAM_MNIST_final_loss_2d
evgo=EVGO_MNIST_final_loss_2d
cm= CM_MNIST_final_loss_2d
rms= RmsProp_MNIST_final_loss_2d
adag= Adagrad_MNIST_final_loss_2d

opac=0.6

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

k=9
x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y
zs = np.array(gd)
Z_gd = zs.reshape(X.shape)

zs = np.array(adam)
Z_adam= zs.reshape(X.shape)

zs = np.array(evgo)
Z_evgo = zs.reshape(X.shape)

zs = np.array(cm)
Z_cm = zs.reshape(X.shape)

zs = np.array(rms)
Z_rms = zs.reshape(X.shape)

zs = np.array(adag)
Z_adag = zs.reshape(X.shape)

#ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.plot_surface(X[:,5:], Y[:,5:], Z_gd[:,5:], color='red', alpha=opac, label="GD")
ax.plot_surface(X[:,5:], Y[:,5:], Z_adam[:,5:], color='lightgreen', alpha=opac)
ax.plot_surface(X[:,5:], Y[:,5:], Z_cm[:,5:], color='black', alpha=opac)
ax.plot_surface(X[:,5:], Y[:,5:], Z_rms[:,5:], color='purple', alpha=opac)
ax.plot_surface(X[:,5:], Y[:,5:], Z_adag[:,5:], color='y', alpha=opac)
ax.plot_surface(X[:,5:], Y[:,5:], Z_evgo[:,5:], color='b', alpha=opac)
fig.suptitle("the Loss surfaces for MNIST")


ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('Loss')
ax.margins(x=0, y=0, z=0.9)
#plt.savefig("f_2_fonksiyonu_r3te_y_yegore_turev.png", dpi=600)
ax.view_init(azim = 100, elev = 0)

fake2Dline = mpl.lines.Line2D([0],[0], linestyle="-", c='red')
fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="-", c='lightgreen')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="-", c='y')
fake2Dline3 = mpl.lines.Line2D([0],[0], linestyle="-", c='black')
fake2Dline4 = mpl.lines.Line2D([0],[0], linestyle="-", c='purple')
fake2Dline5 = mpl.lines.Line2D([0],[0], linestyle="-", c='b')
ax.legend([fake2Dline, fake2Dline1, fake2Dline2,fake2Dline3,fake2Dline4,fake2Dline5], ['GD', 'Adam', 'CM', 'RmsProp', 'Adagrad',  'EVGO'], numpoints = 1)
ax.set_zlim(-3,12)

plt.savefig("the Loss surfaces for MNIST2.png", dpi=600)
plt.show()


# In[208]:



gd=GD_MNIST_final_acc_2d*100
adam= ADAM_MNIST_final_acc_2d*100
evgo=EVGO_MNIST_final_acc_2d*100
cm= CM_MNIST_final_acc_2d*100
rms= RmsProp_MNIST_final_acc_2d*100
adag= Adagrad_MNIST_final_acc_2d*100

opac=0.6

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

k=9
x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y
zs = np.array(gd)
Z_gd = zs.reshape(X.shape)

zs = np.array(adam)
Z_adam= zs.reshape(X.shape)

zs = np.array(evgo)
Z_evgo = zs.reshape(X.shape)

zs = np.array(cm)
Z_cm = zs.reshape(X.shape)

zs = np.array(rms)
Z_rms = zs.reshape(X.shape)

zs = np.array(adag)
Z_adag = zs.reshape(X.shape)

#ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.plot_surface(X, Y, Z_gd, color='red', alpha=opac, label="GD")
ax.plot_surface(X, Y, Z_adam, color='lightgreen', alpha=opac)
ax.plot_surface(X, Y, Z_cm, color='black', alpha=opac)
ax.plot_surface(X, Y, Z_rms, color='purple', alpha=opac)
ax.plot_surface(X, Y, Z_adag, color='y', alpha=opac)
ax.plot_surface(X, Y, Z_evgo, color='b', alpha=opac)
fig.suptitle("the Accuracy surfaces for MNIST")


ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('Accuracy')
ax.margins(x=0, y=0, z=0.9)
#plt.savefig("f_2_fonksiyonu_r3te_y_yegore_turev.png", dpi=600)
ax.view_init(azim = 100, elev = 0)

fake2Dline = mpl.lines.Line2D([0],[0], linestyle="-", c='red')
fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="-", c='lightgreen')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="-", c='y')
fake2Dline3 = mpl.lines.Line2D([0],[0], linestyle="-", c='black')
fake2Dline4 = mpl.lines.Line2D([0],[0], linestyle="-", c='purple')
fake2Dline5 = mpl.lines.Line2D([0],[0], linestyle="-", c='b')
ax.legend([fake2Dline, fake2Dline1, fake2Dline2,fake2Dline3,fake2Dline4,fake2Dline5], ['GD', 'Adam', 'CM', 'RmsProp', 'Adagrad',  'EVGO'], numpoints = 1)
ax.set_zlim(0,100)

plt.savefig("the Accuracy surfaces for MNIST.png", dpi=600)
plt.show()


# In[209]:


gd=GD_MNIST_final_acc_2d*100
adam= ADAM_MNIST_final_acc_2d*100
evgo=EVGO_MNIST_final_acc_2d*100
cm= CM_MNIST_final_acc_2d*100
rms= RmsProp_MNIST_final_acc_2d*100
adag= Adagrad_MNIST_final_acc_2d*100

opac=0.6

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

k=9
x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y
zs = np.array(gd)
Z_gd = zs.reshape(X.shape)

zs = np.array(adam)
Z_adam= zs.reshape(X.shape)

zs = np.array(evgo)
Z_evgo = zs.reshape(X.shape)

zs = np.array(cm)
Z_cm = zs.reshape(X.shape)

zs = np.array(rms)
Z_rms = zs.reshape(X.shape)

zs = np.array(adag)
Z_adag = zs.reshape(X.shape)

#ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.plot_surface(X[:,5:], Y[:,5:], Z_gd[:,5:], color='red', alpha=opac, label="GD")
ax.plot_surface(X[:,5:], Y[:,5:], Z_adam[:,5:], color='lightgreen', alpha=opac)
ax.plot_surface(X[:,5:], Y[:,5:], Z_cm[:,5:], color='black', alpha=opac)
ax.plot_surface(X[:,5:], Y[:,5:], Z_rms[:,5:], color='purple', alpha=opac)
ax.plot_surface(X[:,5:], Y[:,5:], Z_adag[:,5:], color='y', alpha=opac)
ax.plot_surface(X[:,5:], Y[:,5:], Z_evgo[:,5:], color='b', alpha=opac)
fig.suptitle("the Accuracy surfaces for MNIST")


ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('Accuracy')
ax.margins(x=0, y=0, z=0.9)
#plt.savefig("f_2_fonksiyonu_r3te_y_yegore_turev.png", dpi=600)
ax.view_init(azim = 100, elev = 0)

fake2Dline = mpl.lines.Line2D([0],[0], linestyle="-", c='red')
fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="-", c='lightgreen')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="-", c='y')
fake2Dline3 = mpl.lines.Line2D([0],[0], linestyle="-", c='black')
fake2Dline4 = mpl.lines.Line2D([0],[0], linestyle="-", c='purple')
fake2Dline5 = mpl.lines.Line2D([0],[0], linestyle="-", c='b')
ax.legend([fake2Dline, fake2Dline1, fake2Dline2,fake2Dline3,fake2Dline4,fake2Dline5], ['GD', 'Adam', 'CM', 'RmsProp', 'Adagrad',  'EVGO'], numpoints = 1)
ax.set_zlim(0,100)

plt.savefig("the Accuracy surfaces for MNIST2.png", dpi=600)
plt.show()


# In[211]:



colorize='jet'

lw=5

k=9


gd=GD_MNIST_final_loss_2d
adam= ADAM_MNIST_final_loss_2d
evgo=EVGO_MNIST_final_loss_2d
cmx= CM_MNIST_final_loss_2d
rms= RmsProp_MNIST_final_loss_2d
adag= Adagrad_MNIST_final_loss_2d


x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y

fig = plt.figure(figsize=(27, 8))

plt.subplot(231)
zs = np.array(gd)
Z = zs.reshape(X.shape)

cp = plt.contour(X[:,5:], Y[:,5:], Z[:,5:] , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Loss surface for MNIST using GD')

plt.subplot(232)
zs = np.array(adam)
Z = zs.reshape(X.shape)

cp = plt.contour(X[:,5:], Y[:,5:], Z[:,5:], linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Loss surface for MNIST using Adam')
plt.subplot(236)
zs = np.array(evgo)
Z = zs.reshape(X.shape)

cp = plt.contour(X[:,5:], Y[:,5:], Z[:,5:], linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)

plt.title('The contour of the Loss surface for MNIST using EVGO')

plt.subplot(235)
zs = np.array(adag)
Z = zs.reshape(X.shape)

cp = plt.contour(X[:,5:], Y[:,5:], Z[:,5:] , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Loss surface for MNIST using Adagrad')

plt.subplot(233)
zs = np.array(cmx)
Z = zs.reshape(X.shape)

cp = plt.contour(X[:,5:], Y[:,5:], Z[:,5:] , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Loss surface for MNIST using CM')
plt.subplot(234)
zs = np.array(rms)
Z = zs.reshape(X.shape)

cp = plt.contour(X[:,5:], Y[:,5:], Z[:,5:] , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)

plt.title('The contour of the Loss surface for MNIST using RmsProp')

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.015, 0.8])

cmap = mpl.cm.jet
bounds = [0,1,2,3,4,5,6,7,8,8,10,11,12,13,14]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#plt.colorbar(cax=cax, extend='both', cmap=colorize)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colorize), cax=cax)
plt.savefig("xThe contour of the Loss surface for MNIST.png", dpi=600)
plt.show()

