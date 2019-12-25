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


# In[2]:




D_final_acc_EVGO = np.loadtxt(open("D_final_acc_EVGO.csv", "rb"), delimiter=",")
D_final_acc_EVGO_2d = np.loadtxt(open("D_final_acc_EVGO_2d.csv", "rb"), delimiter=",")

D_final_loss_EVGO = np.loadtxt(open("D_final_loss_EVGO.csv", "rb"), delimiter=",")
D_final_loss_EVGO_2d = np.loadtxt(open("D_final_loss_EVGO_2d.csv", "rb"), delimiter=",")

D_lossepoch_mean_EVGO = np.loadtxt(open("D_lossepoch_mean_EVGO.csv", "rb"), delimiter=",")
D_lossepoch_mean_EVGO_cin = np.loadtxt(open("D_lossepoch_mean_EVGO_cin.csv", "rb"), delimiter=",")
D_lossepoch_mean_EVGO_cip = np.loadtxt(open("D_lossepoch_mean_EVGO_cip.csv", "rb"), delimiter=",")
        



# In[3]:


D_final_acc_Adam = np.loadtxt(open("D_final_acc_Adam.csv", "rb"), delimiter=",")
D_final_acc_Adam_2d = np.loadtxt(open("D_final_acc_Adam_2d.csv", "rb"), delimiter=",")

D_final_loss_Adam = np.loadtxt(open("D_final_loss_Adam.csv", "rb"), delimiter=",")
D_final_loss_Adam_2d = np.loadtxt(open("D_final_loss_Adam_2d.csv", "rb"), delimiter=",")

D_lossepoch_mean_Adam = np.loadtxt(open("D_lossepoch_mean_Adam.csv", "rb"), delimiter=",")
D_lossepoch_mean_Adam_cin = np.loadtxt(open("D_lossepoch_mean_Adam_cin.csv", "rb"), delimiter=",")
D_lossepoch_mean_Adam_cip = np.loadtxt(open("D_lossepoch_mean_Adam_cip.csv", "rb"), delimiter=",")


# In[4]:


D_final_acc_GD = np.loadtxt(open("D_final_acc_GD.csv", "rb"), delimiter=",")
D_final_acc_GD_2d = np.loadtxt(open("D_final_acc_GD_2d.csv", "rb"), delimiter=",")

D_final_loss_GD = np.loadtxt(open("D_final_loss_GD.csv", "rb"), delimiter=",")
D_final_loss_GD_2d = np.loadtxt(open("D_final_loss_GD_2d.csv", "rb"), delimiter=",")

D_lossepoch_mean_GD = np.loadtxt(open("D_lossepoch_mean_GD.csv", "rb"), delimiter=",")
D_lossepoch_mean_GD_cin = np.loadtxt(open("D_lossepoch_mean_GD_cin.csv", "rb"), delimiter=",")
D_lossepoch_mean_GD_cip = np.loadtxt(open("D_lossepoch_mean_GD_cip.csv", "rb"), delimiter=",")


# In[5]:


alfa=np.arange(0, 1., 0.02)
beta=np.arange(0, 1., 0.02)
alfa.shape

fig = plt.figure(figsize=(15, 8))


for i in range(D_final_loss_Adam.shape[0]):
    plt.plot(alfa, D_final_loss_GD[i], color='red')
    plt.plot(alfa, D_final_loss_Adam[i], color='darkgreen')
    plt.plot(alfa, D_final_loss_EVGO[i], color='darkblue')

plt.legend(['GD', 'Adam', 'EVGO'], loc='upper right')

plt.xlabel("alpha")
plt.ylabel("Loss")
plt.yticks(np.arange(0, 9, 2.0))

plt.tight_layout()

plt.savefig("1d_loss_Logistic Regression for D.png", dpi=600)

plt.show()


# In[6]:


fig = plt.figure(figsize=(15, 8))


for i in range(D_final_acc_Adam.shape[0]):
    plt.plot(alfa, D_final_acc_GD[i]*100, color='red')
    plt.plot(alfa, D_final_acc_Adam[i]*100, color='darkgreen')
    plt.plot(alfa, D_final_acc_EVGO[i]*100, color='darkblue')

plt.legend(['GD', 'Adam', 'EVGO'], loc='upper right')

plt.xlabel("alpha")
plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("1d_acc_Logistic Regression for D.png", dpi=600)

plt.show()


# In[9]:


data_evgo = np.average(D_final_loss_EVGO_2d, axis=0)
data_gd = np.average(D_final_loss_GD_2d, axis=0)
data_adam = np.average(D_final_loss_Adam_2d, axis=0)

opac=0.7

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

k=10
x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y
zs = np.array(data_gd)
Z_gd = zs.reshape(X.shape)

zs = np.array(data_adam)
Z_adam= zs.reshape(X.shape)

zs = np.array(data_evgo)
Z_evgo = zs.reshape(X.shape)

#ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.plot_surface(X, Y, Z_gd, color='red', alpha=opac, label="GD")
ax.plot_surface(X, Y, Z_adam, color='lightgreen', alpha=opac)
ax.plot_surface(X, Y, Z_evgo, color='b', alpha=opac)

fig.suptitle("the Loss surface for Data $D$")


ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('Loss')
ax.margins(x=0, y=0, z=0.9)
#plt.savefig("f_2_fonksiyonu_r3te_y_yegore_turev.png", dpi=600)
ax.view_init(azim = 220, elev = 30)

fake2Dline = mpl.lines.Line2D([0],[0], linestyle="-", c='red')
fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="-", c='lightgreen')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="-", c='b')
ax.legend([fake2Dline, fake2Dline1, fake2Dline2], ['GD', 'Adam', 'EVGO'], numpoints = 1)
ax.set_zlim(-3,10)

plt.savefig("the Loss surface for Data D.png", dpi=600)
plt.show()


# In[10]:


data_evgo = np.average(D_final_acc_EVGO_2d, axis=0)*100
data_gd = np.average(D_final_acc_GD_2d, axis=0)*100
data_adam = np.average(D_final_acc_Adam_2d, axis=0)*100

opac=0.7

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

k=10
x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y
zs = np.array(data_gd)
Z_gd = zs.reshape(X.shape)

zs = np.array(data_adam)
Z_adam= zs.reshape(X.shape)

zs = np.array(data_evgo)
Z_evgo = zs.reshape(X.shape)

#ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.plot_surface(X, Y, Z_gd, color='red', alpha=opac, label="GD")
ax.plot_surface(X, Y, Z_adam, color='lightgreen', alpha=opac)
ax.plot_surface(X, Y, Z_evgo, color='b', alpha=opac)

fig.suptitle("the Accuracy surface for Data $D$")


ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('Accuracy')
ax.margins(x=0, y=0, z=0.9)
#plt.savefig("f_2_fonksiyonu_r3te_y_yegore_turev.png", dpi=600)
ax.view_init(azim = 220, elev = 30)

fake2Dline = mpl.lines.Line2D([0],[0], linestyle="-", c='red')
fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="-", c='lightgreen')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="-", c='b')
ax.legend([fake2Dline, fake2Dline1, fake2Dline2], ['GD', 'Adam', 'EVGO'], numpoints = 1)
ax.set_zlim(0,100)

plt.savefig("the Accuracy surface for Data D.png", dpi=600)
plt.show()


# In[11]:


data_evgo = np.average(D_final_loss_EVGO_2d, axis=0)
data_gd = np.average(D_final_loss_GD_2d, axis=0)
data_adam = np.average(D_final_loss_Adam_2d, axis=0)


k=10
x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y
zs = np.array(data_evgo)
Z = zs.reshape(X.shape)
cp = plt.contour(X, Y, Z , linewidths=3, cmap='jet')
plt.clabel(cp, inline=True, 
          fontsize=10)

    
plt.colorbar(cp, shrink=0.8, extend='both', cmap='jet')

    
plt.title('The contour of the Loss surface for D using EVGO')
plt.flag()

plt.xlabel('alpha')
plt.ylabel('beta')
plt.clim(0,2)
plt.savefig("The contour of the Loss surface for D using EVGO.png", dpi=600)
plt.show()


# In[12]:


data_evgo = np.average(D_final_loss_EVGO_2d, axis=0)
data_gd = np.average(D_final_loss_GD_2d, axis=0)
data_adam = np.average(D_final_loss_Adam_2d, axis=0)


k=10
x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y
zs = np.array(data_gd)
Z = zs.reshape(X.shape)
cp = plt.contour(X, Y, Z , linewidths=3, cmap='jet')
plt.clabel(cp, inline=True, 
          fontsize=10)

    
plt.colorbar(cp, shrink=0.8, extend='both', cmap='jet')

    
plt.title('The contour of the Loss surface for D using GD')
plt.flag()

plt.xlabel('alpha')
plt.ylabel('beta')
plt.clim(0,2)
plt.savefig("The contour of the Loss surface for D using GD.png", dpi=600)
plt.show()


# In[13]:




colorize='jet'

lw=5

k=10


data_evgo = np.average(D_final_loss_EVGO_2d, axis=0)
data_gd = np.average(D_final_loss_GD_2d, axis=0)
data_adam = np.average(D_final_loss_Adam_2d, axis=0)


x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y

fig = plt.figure(figsize=(27, 8))

plt.subplot(131)
zs = np.array(data_gd)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Loss surface for D using GD')

plt.subplot(132)
zs = np.array(data_adam)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Loss surface for D using Adam')
plt.subplot(133)
zs = np.array(data_evgo)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)

plt.title('The contour of the Loss surface for D using EVGO')

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.015, 0.8])
plt.colorbar(cax=cax, extend='both', cmap=colorize)
plt.savefig("The contour of the Loss surface for D.png", dpi=600)
plt.show()


# In[14]:




colorize='jet'

lw=5

k=10


data_evgo = np.average(D_final_acc_EVGO_2d, axis=0)*100
data_gd = np.average(D_final_acc_GD_2d, axis=0)*100
data_adam = np.average(D_final_acc_Adam_2d, axis=0)*100


x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y

fig = plt.figure(figsize=(27, 8))

plt.subplot(131)
zs = np.array(data_gd)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Accuracy surface for D using GD')

plt.subplot(132)
zs = np.array(data_adam)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Accuracy surface for D using Adam')
plt.subplot(133)
zs = np.array(data_evgo)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)

plt.title('The contour of the Accuracy surface for D using EVGO')

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.015, 0.8])
plt.colorbar(cax=cax, extend='both', cmap=colorize)
plt.savefig("The contour of the Accuracy surface for D.png", dpi=600)
plt.show()


# In[15]:


np.sort(data_evgo)


# In[16]:


np.sort(data_adam)


# In[17]:


np.sort(data_gd)


# In[7]:




colorize='jet'

lw=5

k=10


data_evgo = np.average(D_final_loss_EVGO_2d, axis=0)
data_gd = np.average(D_final_loss_GD_2d, axis=0)
data_adam = np.average(D_final_loss_Adam_2d, axis=0)


x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y

fig = plt.figure(figsize=(27, 8))

plt.subplot(131)
zs = np.array(data_gd)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Loss surface for D using GD')

plt.subplot(132)
zs = np.array(data_adam)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Loss surface for D using Adam')
plt.subplot(133)
zs = np.array(data_evgo)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)

plt.title('The contour of the Loss surface for D using EVGO')

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.015, 0.8])
cmap = mpl.cm.jet
bounds = [0,1,2,3,4,5,6,7,8,8,10,11,12,13,14]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#plt.colorbar(cax=cax, extend='both', cmap=colorize)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colorize), cax=cax)
plt.savefig("xThe contour of the Loss surface for D.png", dpi=600)
plt.show()


# In[6]:




colorize='jet'

lw=5

k=10


data_evgo = np.average(D_final_acc_EVGO_2d, axis=0)*100
data_gd = np.average(D_final_acc_GD_2d, axis=0)*100
data_adam = np.average(D_final_acc_Adam_2d, axis=0)*100


x = y = np.linspace(-2, 2, k)
X, Y = np.meshgrid(x, y)
X, Y

fig = plt.figure(figsize=(27, 8))

plt.subplot(131)
zs = np.array(data_gd)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Accuracy surface for D using GD')

plt.subplot(132)
zs = np.array(data_adam)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.title('The contour of the Accuracy surface for D using Adam')
plt.subplot(133)
zs = np.array(data_evgo)
Z = zs.reshape(X.shape)

cp = plt.contour(X, Y, Z , linewidths=lw, cmap=colorize)
plt.clabel(cp, inline=True, 
          fontsize=10)

plt.title('The contour of the Accuracy surface for D using EVGO')

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.015, 0.8])
cmap = mpl.cm.jet
bounds = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100, 105]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#plt.colorbar(cax=cax, extend='both', cmap=colorize)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colorize), cax=cax)
plt.savefig("xThe contour of the Accuracy surface for D.png", dpi=600)
plt.show()


# In[ ]:




