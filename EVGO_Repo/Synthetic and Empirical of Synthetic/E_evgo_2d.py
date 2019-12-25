#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras import backend as K
import os
import keras
import pickle
import os.path
import numpy as np
import tensorflow as tf
import numpy
import scipy.stats
import numpy as np



seed_value= 0

import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
import os
import keras
import pickle
import os.path
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelBinarizer
from keras import initializers
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit


# In[2]:


rng= numpy.random.RandomState(25)
N = 400                                   # training sample size
feats = 784

D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

x_train = D[0]
y_train = keras.utils.to_categorical(D[1], 2)


def build_model(SS):
    """
    Builds test Keras model for LeNet MNIST
    :param loss (str): Type of loss - must be one of Keras accepted keras losses
    :return: Keras dense model of predefined structure
    """
    input = Input(shape=(feats,))
    dens1=Dense(units=2, activation='softmax')(input)
    model = Model(input=input, output=dens1)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model

def Run(seed):  
    
    all_model = [None,None,None]
    losses = [None,None,None]

    prediction=[]

    all_score =[0,0,0]
    gr=[]
    wr=[]
    xwr=[]

    for i in range(3):
        np.random.seed(seed+i)
        model = build_model(seed+i)
        all_model[i]=model

    for i in range(3):    
        weights = all_model[i].trainable_weights # weight tensors
        weights = [weight for weight in weights] # filter down weights tensors to only ones which are trainable
        gradients = all_model[i].optimizer.get_gradients(all_model[i].total_loss, weights) # gradient tensors
        gr.append(gradients)
        wr.append(weights)
        xweights = all_model[i].non_trainable_weights # weight tensors
        xweights = [weight for weight in xweights] # filter down weights tensors to only ones which are trainable
        xwr.append(xweights)

        losses[i]=all_model[i].total_loss
        prediction.append(all_model[i].output)
        
    
    input_tensors = [all_model[0].inputs[0], # input data
                 all_model[0].sample_weights[0], # how much to weight each sample by
                 all_model[0].targets[0], # labels
                 K.learning_phase(), # train or test mode
                 all_model[1].inputs[0], # input data
                 all_model[1].sample_weights[0], # how much to weight each sample by
                 all_model[1].targets[0], # labels
                 all_model[2].inputs[0], # input data
                 all_model[2].sample_weights[0], # how much to weight each sample by
                 all_model[2].targets[0], # labels
                ]


    minlos = K.argmin(losses)

    grr=[]
    for x in gr:
        for y in x:
            grr.append(y)

    upd_test= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2], minlos, prediction[0], prediction[1], prediction[2] ])


    grad_best=[]
    grad_non0 = []
    grad_non1 = []


    weig_best=[]
    weig_non0 = []
    weig_non1 = []

    xweig_best=[]
    xweig_non0 = []
    xweig_non1 = []




    for i in range(len(gr[0])):
        gr_ck=tf.concat([gr[0][i],gr[1][i], gr[2][i]],0)
        newshape = (3, ) + (tuple(wr[0][i].shape))


        gr_ck2=tf.reshape(gr_ck, newshape)

        bb = gr_ck2[minlos]
        grad_best.append(bb)

        nbb0 = gr_ck2[0:minlos]                       #[0,enk) U (enk,] aralıklarının birleşimi bize nonbesti verecek
        nbb1 = gr_ck2[minlos+1:]                      #[0,enk) U (enk,] aralıklarının birleşimi bize nonbesti verecek
        nbc = tf.concat([nbb0,nbb1], 0)    
        nbc = tf.reshape(nbc, (-1,))
        newshape2 = (2, ) + (tuple(wr[0][i].shape))

        nbc2 = tf.reshape(nbc, newshape2) 
        nb0 = nbc2[0]
        nb1 = nbc2[1]
        grad_non0.append(nb0)
        grad_non1.append(nb1)


        wr_ck=tf.concat([wr[0][i],wr[1][i], wr[2][i]],0)

        newshape = (3, ) + (tuple(wr[0][i].shape))
        wr_ck2=tf.reshape(wr_ck, newshape) 
        bb2 = wr_ck2[minlos]
        weig_best.append(bb2)

        #wb = wr_ck[minlos]
        wnbb0 = wr_ck2[0:minlos]                       #[0,enk) U (enk,] aralıklarının birleşimi bize nonbesti verecek
        wnbb1 = wr_ck2[minlos+1:]                      #[0,enk) U (enk,] aralıklarının birleşimi bize nonbesti verecek
        wnbc = tf.concat([wnbb0,wnbb1],0)    
        wnbc = tf.reshape(wnbc, (-1,))
        newshape2 = (2, ) + (tuple(wr[0][i].shape))

        wnbc2 =tf.reshape(wnbc, newshape2)
        wnb0 = wnbc2[0]
        wnb1 = wnbc2[1]
        weig_non0.append(wnb0)
        weig_non1.append(wnb1)

        if i<len(xwr[0]):
            print (i)
            xwr_ck=tf.concat([xwr[0][i],xwr[1][i], xwr[2][i]], 0)

            newshape = (3, ) + (tuple(xwr[0][i].shape))

            xwr_ck2=tf.reshape(xwr_ck, newshape)  
            xbb2 = xwr_ck2[minlos]
            xweig_best.append(xbb2)

            #wb = wr_ck[minlos]
            xwnbb0 = xwr_ck2[0:minlos]                       #[0,enk) U (enk,] aralıklarının birleşimi bize nonbesti verecek
            xwnbb1 = xwr_ck2[minlos+1:]                      #[0,enk) U (enk,] aralıklarının birleşimi bize nonbesti verecek
            xwnbc = tf.concat([xwnbb0,xwnbb1], 0)    

            xwnbc = tf.reshape(xwnbc, (-1,))
            newshape2 = (2, ) + (tuple(xwr[0][i].shape))

            xwnbc2 = tf.reshape(xwnbc, newshape2) 
            xwnb0 = xwnbc2[0]
            xwnb1 = xwnbc2[1]
            xweig_non0.append(xwnb0)
            xweig_non1.append(xwnb1)
        else:
            pass


    los=tf.stack([losses[0], losses[1], losses[2]])

    newshape = (3, )
    los2=tf.reshape(los, newshape) 
    losbest = los2[minlos]

    #wb = wr_ck[minlos]
    los_0 = los2[0:minlos]                       #[0,enk) U (enk,] aralıklarının birleşimi bize nonbesti verecek
    los_1 = los2[minlos+1:]                      #[0,enk) U (enk,] aralıklarının birleşimi bize nonbesti verecek
    loswnbc = tf.concat([los_0,los_1],0)    
    loswnbc = tf.reshape(loswnbc,(-1,))
    newshape2 = (2, )

    loswnbc2 = tf.reshape(loswnbc, newshape2)
    losss0 = loswnbc2[0]
    losss1 = loswnbc2[1]



    lr = 0.1
    lr2 = 0.02
    lr3 = 0.7
    eps = 0.1


    mn0 = [tf.keras.backend.l2_normalize((best-nonbest)*(losbest-losss0)/ tf.reduce_sum(tf.pow((best-nonbest),2)+eps))  for best, nonbest in zip(weig_best, weig_non0)]

    mn1 = [tf.keras.backend.l2_normalize((best-nonbest)*(losbest-losss1)/ tf.reduce_sum(tf.pow((best-nonbest),2)+eps))  for best, nonbest in zip(weig_best, weig_non1)]



    nCom0 = [non- lr3* grad - lr2* mn for mn, grad, non in zip(mn0,grad_non0, weig_non0 )]

    nCom1 = [non- lr3* grad - lr2 * mn for mn, grad, non in zip(mn1,grad_non1, weig_non1 )]

    xbest = [ -lr * nc + non for nc, non in zip(grad_best, weig_best)]


    upd2 = [
        tf.assign(param_i, v)
        for param_i, v in zip(wr[2], xbest)
    ]

    upd2.extend(
            [tf.assign(param_i, v)
            for param_i, v in zip(xwr[2], xweig_best)]
        )

    upd2.extend(
            [tf.assign(param_i, v)
            for param_i, v in zip(wr[1], nCom0)]
        )
    upd2.extend(
            [tf.assign(param_i, v)
            for param_i, v in zip(xwr[1], xweig_non0)]
        )
    upd2.extend(
            [tf.assign(param_i, v)
            for param_i, v in zip(wr[0], nCom1)]
        )
    upd2.extend(
            [tf.assign(param_i, v)
            for param_i, v in zip(xwr[0], xweig_non1)]
        )
    

    upd_bb2= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2], minlos, prediction[0], prediction[1], prediction[2], wr[2][0], wr[2][1]], updates=upd2)
    #upd_bb3= K.function(inputs=input_tensors, outputs= [allw[0], allw[1]])
    allw=all_model[2].get_weights()
    
    return (upd_bb2)

inputs = [x_train, # X
  np.ones(y_train.shape[0]), # sample weights
  y_train, # y
  1, # learning phase in Train mode
  x_train, # X
  np.ones(y_train.shape[0]), # sample weights
  y_train, # y
  x_train, # X
  np.ones(y_train.shape[0]), # sample weights
  y_train, # y
         ]


# In[3]:


lossepoch=[]
acctra=[]



loss_val=[]
acc_val=[]
lossepoch_val=[]
epochs=100

optim=[]
init=[]

optim_b=[]
init_b=[]



for ru in range(30):
    print ('run ' + str(ru))
    upd_func = Run(ru)
    
    lossHistory = []
    acc_hist = []

    
    for f in tqdm(range(epochs)):
        program_starts = time.time()

        
        print('Epoch', f)
        print ('train')

        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        ll = upd_func(inputs)

        
        yhat=ll[6]  
        lossHistory.append(ll[2])
        print ('train loss score is :'+str(ll[2]))
        print ('train acc score is :'+str(accuracy_score(np.argmax(y_train,axis=1), np.argmax(yhat,axis=1))))
        acctra.append(accuracy_score(np.argmax(y_train,axis=1), np.argmax(yhat,axis=1)))
        if f==epochs-1:
            optim.append(ll[-2])
            optim_b.append(ll[-1])
            print ("optim")
        elif f==0:
            print ("init")
            init.append(ll[-2])
            init_b.append(ll[-1])
        else:
            pass
        now = time.time()
        print("It has been {0} seconds since the loop started".format(now - program_starts))
    lossepoch.append(lossHistory)
    acctra.append(acc_hist)
    K.clear_session()


# In[4]:


def surface_2d_inp_X(ini, opt):
    alfa=np.linspace(-2, 2, 10)
    beta=np.linspace(-2, 2, 10)
    ALFA, BETA = np.meshgrid(alfa, beta)
 
    
    results= []
    
    for (i,j,k) in zip(np.ravel(ALFA), np.ravel(BETA), range(100)):
        k=k+1
        numpy.random.seed(k)       
        teta=np.random.rand(ini.shape[0], ini.shape[1])
        d1=opt-teta
        d2=ini-teta      
        

        fi = i*d1+teta 
        ro = i*d2+teta 
        
        res = j*fi+(1-j)*ro
        results.append(res)
    
    return results


# In[5]:


def surface_2d_inp_b(ini, opt):
    alfa=np.linspace(-2, 2, 10)
    beta=np.linspace(-2, 2, 10)
    ALFA, BETA = np.meshgrid(alfa, beta)
 
    
    results= []
    
    for (i,j,k) in zip(np.ravel(ALFA), np.ravel(BETA), range(100)):
        k=k+1
        numpy.random.seed(k)       
        teta=np.random.rand(ini.shape[0],)
        d1=opt-teta
        d2=ini-teta      
        

        fi = i*d1+teta 
        ro = i*d2+teta 
        
        res = j*fi+(1-j)*ro
        results.append(res)
    
    return results


# In[6]:


final_loss_2d=[]
final_acc_2d=[]

model_str = build_model(105)

for j in range(30):
    
    res = surface_2d_inp_X(init[j], optim[j])
    res_b = surface_2d_inp_b(init_b[j], optim_b[j])
    fin_loss=[]
    fin_acc=[]
    for k in range(100): 
        model_str.layers[1].set_weights([res[k],res_b[k]])
        loss_f, acc_f = model_str.evaluate(x_train,y_train)
        fin_loss.append(loss_f)
        fin_acc.append(acc_f)
    final_loss_2d.append(fin_loss)
    final_acc_2d.append(fin_acc)


# In[7]:


def surface_1d_inp(ini, opt):
    alfa=np.arange(0, 1.02, 0.02)
    
    results= []
    
    for i in range(50):
        res = ini + alfa[i]*(opt-ini)
        results.append(res)
    
    return results


# In[8]:


final_loss=[]
final_acc=[]

model_str = build_model(105)

for j in range(30):
    
    res = surface_1d_inp(init[j], optim[j])
    res_b = surface_1d_inp(init_b[j], optim_b[j])
    fin_loss=[]
    fin_acc=[]
    for k in range(50): 
        model_str.layers[1].set_weights([res[k],res_b[k]])
        loss_f, acc_f = model_str.evaluate(x_train,y_train)
        fin_loss.append(loss_f)
        fin_acc.append(acc_f)
    final_loss.append(fin_loss)
    final_acc.append(fin_acc)


# In[9]:


np.array(final_loss).shape


# In[10]:


mean_gd=[]
ci_n_gd=[]
ci_p_gd=[]

for i in range(100):
    m, m_n, m_p = mean_confidence_interval(np.array(lossepoch)[:,i])
    mean_gd.append(m)
    ci_n_gd.append(m_n)
    ci_p_gd.append(m_p)


# In[11]:


np.savetxt("E_lossepoch_mean_EVGO.csv", np.array(mean_gd), delimiter=",", fmt='%s')  
np.savetxt("E_lossepoch_mean_EVGO_cin.csv", np.array(ci_n_gd), delimiter=",", fmt='%s')  
np.savetxt("E_lossepoch_mean_EVGO_cip.csv", np.array(ci_p_gd), delimiter=",", fmt='%s')  

np.savetxt("E_final_loss_EVGO.csv", np.array(final_loss), delimiter=",", fmt='%s')
np.savetxt("E_final_acc_EVGO.csv", np.array(final_acc), delimiter=",", fmt='%s')  

np.savetxt("E_final_loss_EVGO_2d.csv", np.array(final_loss_2d), delimiter=",", fmt='%s')
np.savetxt("E_final_acc_EVGO_2d.csv", np.array(final_acc_2d), delimiter=",", fmt='%s')  


# In[ ]:




