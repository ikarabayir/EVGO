#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function

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

dataset='mnist.pkl.gz'
batch_size = 128
num_classes = 10
epochs = 100
lr = 0.1
lr2 = 0.02
lr3 = 0.7
eps = 0.1
#K.clear_session()


print('... loading data')

# Load the dataset
with gzip.open(dataset, 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)
        
        
(x_train, y_train) = train_set
(x_test, y_test) = test_set
(x_val, y_val) = valid_set
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_val = np.pad(x_val, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape

input_shape = (32, 32, 1)


# In[2]:


import numpy

def surface_1d_inp(ini, opt):
    alfa=np.arange(0, 1.02, 0.02)
    
    results= []
    
    for i in range(50):
        res = ini + alfa[i]*(opt-ini)
        results.append(res)
    
    return results


def surface_2d_inp_C(ini, opt):
    alfa=np.linspace(-2, 2, 9)
    beta=np.linspace(-2, 2, 9)
    ALFA, BETA = np.meshgrid(alfa, beta)
 
    
    results= []
    
    for (i,j,k) in zip(np.ravel(ALFA), np.ravel(BETA), range(81)):
        k=k+1
        numpy.random.seed(k)       
        teta=np.random.rand(ini.shape[0], ini.shape[1], ini.shape[2], ini.shape[3])
        d1=opt-teta
        d2=ini-teta      
        

        fi = i*d1+teta 
        ro = i*d2+teta 
        
        res = j*fi+(1-j)*ro
        results.append(res)
    
    return results

def surface_2d_inp_X(ini, opt):
    alfa=np.linspace(-2, 2, 9)
    beta=np.linspace(-2, 2, 9)
    ALFA, BETA = np.meshgrid(alfa, beta)
 
    
    results= []
    
    for (i,j,k) in zip(np.ravel(ALFA), np.ravel(BETA), range(81)):
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

def surface_2d_inp_b(ini, opt):
    alfa=np.linspace(-2, 2, 9)
    beta=np.linspace(-2, 2, 9)
    ALFA, BETA = np.meshgrid(alfa, beta)
 
    
    results= []
    
    for (i,j,k) in zip(np.ravel(ALFA), np.ravel(BETA), range(81)):
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



# #   EVGO 

# In[3]:


def build_model(setseed):
    """
    Builds test Keras model for LeNet MNIST
    :param loss (str): Type of loss - must be one of Keras accepted keras losses
    :return: Keras dense model of predefined structure
    """
    input = Input(shape=input_shape)
    conv1 = Conv2D(6, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(input)
    avg1 = AveragePooling2D()(conv1)
    conv2 = Conv2D(16, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(avg1)
    avg2 = AveragePooling2D()(conv2)
    flat= Flatten()(avg2)
    dens1=Dense(units=120, activation='relu')(flat)
    dens2=Dense(units=84, activation='relu')(dens1)
    probs=Dense(num_classes, activation='softmax')(dens2)
    
    model = Model(input=input, output=probs)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

all_model = [None,None,None]
losses = [None,None,None]

prediction=[]

all_score =[0,0,0]
gr=[]
wr=[]
xwr=[]

for i in range(3):
    np.random.seed(25+i)
    model = build_model(i+2)
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
    
model.summary()


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


upd_bb2= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2],
                                                   minlos, prediction[0], prediction[1], prediction[2],
                                                  wr[2][0],wr[2][1],wr[2][2],wr[2][3],wr[2][4],
                                                  wr[2][5],wr[2][6],wr[2][7],wr[2][8],wr[2][9]], updates=upd2)



epochs=100 # degistir

lossepoch=[]
lossepoch_test=[]
lossx=[]
acctra=[]
loss_test=[]
acc_test=[]
skip=[]


loss_val=[]
acc_val=[]
lossepoch_val=[]


optim_layer=[]
init_layer=[]

for f in tqdm(range(epochs)):
    program_starts = time.time()
    tr1=[]
    tr2=[]
    res1=[]
    res2=[]
    res3=[]
    res4=[]
    print('Epoch', f)
    print ('train')
    
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    
    batches = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_train, y_train, batch_size=batch_size):
        K.set_learning_phase(1)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in Train mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_bb2(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:4])
        lossepoch.append(ll[2])
        tr1.append(ll[2])
        tr2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        skip.append(ll[3])
        batches += 1
        if batches > len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    m=(len(x_train) / batch_size)-int((len(x_train) / batch_size))
    tr1[-1]*=m
    tr2[-1]*=m
    lossx.append(np.mean(tr1))
    acctra.append(np.mean(tr2))
    print ('train loss score is :'+str(np.mean(tr1)))
    print ('train acc score is :'+str(np.mean(tr2)))
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))

    if f==epochs-1:
        optim_layer.append(ll[7])
        optim_layer.append(ll[8])
        optim_layer.append(ll[9])
        optim_layer.append(ll[10])
        optim_layer.append(ll[11])
        optim_layer.append(ll[12])
        optim_layer.append(ll[13])
        optim_layer.append(ll[14])
        optim_layer.append(ll[15])
        optim_layer.append(ll[16])
        print ("optim")  
    elif f==0:
        init_layer.append(ll[7])
        init_layer.append(ll[8])
        init_layer.append(ll[9])
        init_layer.append(ll[10])
        init_layer.append(ll[11])
        init_layer.append(ll[12])
        init_layer.append(ll[13])
        init_layer.append(ll[14])
        init_layer.append(ll[15])
        init_layer.append(ll[16])
        print ("init_layer")
    else:
        pass

    print ('test')
    batchesx = 0
    
    print ('validation')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in VAl mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_val.append(ll[2])
        res3.append(ll[2])
        res4.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_val) / batch_size:
            break
    m=(len(x_val) / batch_size)-int((len(x_val) / batch_size))
    res3[-1]*=m
    res4[-1]*=m
    loss_val.append(np.mean(res3))
    acc_val.append(np.mean(res4))
    print ('val loss score is :'+str(np.mean(res3)))
    print ('val acc score is :'+str(np.mean(res4)))
    print ('test')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_test, y_test, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in TEST mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_test.append(ll[2])
        res1.append(ll[2])
        res2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_test) / batch_size:
            break
    m=(len(x_test) / batch_size)-int((len(x_test) / batch_size))
    res1[-1]*=m
    res2[-1]*=m
    loss_test.append(np.mean(res1))
    acc_test.append(np.mean(res2))
    print ('test loss score is :'+str(np.mean(res1)))
    print ('test acc score is :'+str(np.mean(res2)))


    
print (min(lossx), np.argmin(lossx))
print (max(acctra), np.argmax(acctra))

print (min(loss_val), np.argmin(loss_val))
print (max(acc_val), np.argmax(acc_val))

print (min(loss_test), np.argmin(loss_test))
print (max(acc_test), np.argmax(acc_test))

print (acc_test[np.argmax(acc_val)], loss_test[np.argmax(acc_val)])
print (acc_test[np.argmin(loss_val)], loss_test[np.argmin(loss_val)])

res=[]

for j in range(10):
    res.append(surface_1d_inp(init_layer[j], optim_layer[j]))
    

final_loss=[]
final_acc=[]

final_loss_val=[]
final_acc_val=[]

final_loss_test=[]
final_acc_test=[]

model_str = build_model(105)

for k in range(50):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss.append(loss_f)
    final_acc.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val.append(loss_f)
    final_acc_val.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test.append(loss_f)
    final_acc_test.append(acc_f)
    
    
final_loss_2d=[]
final_acc_2d=[]

final_loss_val_2d=[]
final_acc_val_2d=[]

final_loss_test_2d=[]
final_acc_test_2d=[]

model_str = build_model(105)

res=[]
res.append(surface_2d_inp_C(init_layer[0], optim_layer[0]))
res.append(surface_2d_inp_b(init_layer[1], optim_layer[1]))
res.append(surface_2d_inp_C(init_layer[2], optim_layer[2]))
res.append(surface_2d_inp_b(init_layer[3], optim_layer[3]))
res.append(surface_2d_inp_X(init_layer[4], optim_layer[4]))
res.append(surface_2d_inp_b(init_layer[5], optim_layer[5]))
res.append(surface_2d_inp_X(init_layer[6], optim_layer[6]))
res.append(surface_2d_inp_b(init_layer[7], optim_layer[7]))
res.append(surface_2d_inp_X(init_layer[8], optim_layer[8]))
res.append(surface_2d_inp_b(init_layer[9], optim_layer[9]))


for k in range(81):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss_2d.append(loss_f)
    final_acc_2d.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val_2d.append(loss_f)
    final_acc_val_2d.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test_2d.append(loss_f)
    final_acc_test_2d.append(acc_f)
    
np.savetxt("EVGO_Mnist_lossepoch.csv", lossepoch, delimiter=",", fmt='%s')
np.savetxt("EVGO_Mnist_loss_tra.csv", lossx, delimiter=",", fmt='%s')
np.savetxt("EVGO_Mnist_acc_tra.csv", acctra, delimiter=",", fmt='%s')
np.savetxt("EVGO_Mnist_loss_test.csv", loss_test, delimiter=",", fmt='%s')
np.savetxt("EVGO_Mnist_acc_test.csv", acc_test, delimiter=",", fmt='%s')
np.savetxt("EVGO_Mnist_loss_val.csv", loss_val, delimiter=",", fmt='%s')
np.savetxt("EVGO_Mnist_acc_val.csv", acc_val, delimiter=",", fmt='%s')

#np.savetxt("xCR2_EVGO3_Mnist_lossepoch_test.csv", lossepoch_test, delimiter=",", fmt='%s')
#np.savetxt("xCR2_EVGO3_Mnist_skip.csv", skip, delimiter=",", fmt='%s')
np.savetxt("EVGO_MNIST_final_loss.csv", final_loss, delimiter=",", fmt='%s')
np.savetxt("EVGO_MNIST_final_acc.csv", final_acc, delimiter=",", fmt='%s')
np.savetxt("EVGO_MNIST_final_loss_val.csv", final_loss_val, delimiter=",", fmt='%s')
np.savetxt("EVGO_MNIST_final_acc_val.csv", final_loss_test, delimiter=",", fmt='%s')
np.savetxt("EVGO_MNIST_final_loss_test.csv", final_acc_val, delimiter=",", fmt='%s')
np.savetxt("EVGO_MNIST_final_acc_test.csv", final_acc_test, delimiter=",", fmt='%s')

np.savetxt("2d_EVGO_MNIST_final_loss.csv", final_loss_2d, delimiter=",", fmt='%s')
np.savetxt("2d_EVGO_MNIST_final_acc.csv", final_acc_2d, delimiter=",", fmt='%s')
np.savetxt("2d_EVGO_MNIST_final_loss_val.csv", final_loss_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_EVGO_MNIST_final_acc_val.csv", final_loss_test_2d, delimiter=",", fmt='%s')
np.savetxt("2d_EVGO_MNIST_final_loss_test.csv", final_acc_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_EVGO_MNIST_final_acc_test.csv", final_acc_test_2d, delimiter=",", fmt='%s')


# In[4]:


K.clear_session()


# #   GD

# In[5]:


def build_model(setseed):
    """
    Builds test Keras model for LeNet MNIST
    :param loss (str): Type of loss - must be one of Keras accepted keras losses
    :return: Keras dense model of predefined structure
    """
    input = Input(shape=input_shape)
    conv1 = Conv2D(6, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(input)
    avg1 = AveragePooling2D()(conv1)
    conv2 = Conv2D(16, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(avg1)
    avg2 = AveragePooling2D()(conv2)
    flat= Flatten()(avg2)
    dens1=Dense(units=120, activation='relu')(flat)
    dens2=Dense(units=84, activation='relu')(dens1)
    probs=Dense(num_classes, activation='softmax')(dens2)
    
    model = Model(input=input, output=probs)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

all_model = [None,None,None]
losses = [None,None,None]

prediction=[]

all_score =[0,0,0]
gr=[]
wr=[]
xwr=[]

for i in range(3):
    np.random.seed(25+i)
    model = build_model(i+2)
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
    
model.summary()


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




lr=0.1

nCom0 = [non- lr * grad for grad, non in zip(grad_non0, weig_non0)]

nCom1 = [non- lr * grad for grad, non in zip(grad_non1, weig_non1)]

xbest = [non -lr * grad for grad, non in zip(grad_best, weig_best)]


# In[15]:


upd2 = [
    tf.assign(param_i, v)
    for param_i, v in zip(wr[2], xbest)
]

upd2.extend(
        [tf.assign(param_i, v)
        for param_i, v in zip(xwr[2], xweig_best)]
    )


upd_bb2= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2],
                                                   minlos, prediction[0], prediction[1], prediction[2],
                                                  wr[2][0],wr[2][1],wr[2][2],wr[2][3],wr[2][4],
                                                  wr[2][5],wr[2][6],wr[2][7],wr[2][8],wr[2][9]], updates=upd2)



epochs=100 # degistir

lossepoch=[]
lossepoch_test=[]
lossx=[]
acctra=[]
loss_test=[]
acc_test=[]
skip=[]


loss_val=[]
acc_val=[]
lossepoch_val=[]


optim_layer=[]
init_layer=[]

for f in tqdm(range(epochs)):
    program_starts = time.time()
    tr1=[]
    tr2=[]
    res1=[]
    res2=[]
    res3=[]
    res4=[]
    print('Epoch', f)
    print ('train')
    
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    
    batches = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_train, y_train, batch_size=batch_size):
        K.set_learning_phase(1)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in Train mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_bb2(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:4])
        lossepoch.append(ll[2])
        tr1.append(ll[2])
        tr2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        skip.append(ll[3])
        batches += 1
        if batches > len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    m=(len(x_train) / batch_size)-int((len(x_train) / batch_size))
    tr1[-1]*=m
    tr2[-1]*=m
    lossx.append(np.mean(tr1))
    acctra.append(np.mean(tr2))
    print ('train loss score is :'+str(np.mean(tr1)))
    print ('train acc score is :'+str(np.mean(tr2)))
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))

    if f==epochs-1:
        optim_layer.append(ll[7])
        optim_layer.append(ll[8])
        optim_layer.append(ll[9])
        optim_layer.append(ll[10])
        optim_layer.append(ll[11])
        optim_layer.append(ll[12])
        optim_layer.append(ll[13])
        optim_layer.append(ll[14])
        optim_layer.append(ll[15])
        optim_layer.append(ll[16])
        print ("optim")  
    elif f==0:
        init_layer.append(ll[7])
        init_layer.append(ll[8])
        init_layer.append(ll[9])
        init_layer.append(ll[10])
        init_layer.append(ll[11])
        init_layer.append(ll[12])
        init_layer.append(ll[13])
        init_layer.append(ll[14])
        init_layer.append(ll[15])
        init_layer.append(ll[16])
        print ("init_layer")
    else:
        pass

    print ('test')
    batchesx = 0
    
    print ('validation')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in VAl mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_val.append(ll[2])
        res3.append(ll[2])
        res4.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_val) / batch_size:
            break
    m=(len(x_val) / batch_size)-int((len(x_val) / batch_size))
    res3[-1]*=m
    res4[-1]*=m
    loss_val.append(np.mean(res3))
    acc_val.append(np.mean(res4))
    print ('val loss score is :'+str(np.mean(res3)))
    print ('val acc score is :'+str(np.mean(res4)))
    print ('test')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_test, y_test, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in TEST mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_test.append(ll[2])
        res1.append(ll[2])
        res2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_test) / batch_size:
            break
    m=(len(x_test) / batch_size)-int((len(x_test) / batch_size))
    res1[-1]*=m
    res2[-1]*=m
    loss_test.append(np.mean(res1))
    acc_test.append(np.mean(res2))
    print ('test loss score is :'+str(np.mean(res1)))
    print ('test acc score is :'+str(np.mean(res2)))


    
print (min(lossx), np.argmin(lossx))
print (max(acctra), np.argmax(acctra))

print (min(loss_val), np.argmin(loss_val))
print (max(acc_val), np.argmax(acc_val))

print (min(loss_test), np.argmin(loss_test))
print (max(acc_test), np.argmax(acc_test))

print (acc_test[np.argmax(acc_val)], loss_test[np.argmax(acc_val)])
print (acc_test[np.argmin(loss_val)], loss_test[np.argmin(loss_val)])


res=[]

for j in range(10):
    res.append(surface_1d_inp(init_layer[j], optim_layer[j]))
    

final_loss=[]
final_acc=[]

final_loss_val=[]
final_acc_val=[]

final_loss_test=[]
final_acc_test=[]

model_str = build_model(105)

for k in range(50):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss.append(loss_f)
    final_acc.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val.append(loss_f)
    final_acc_val.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test.append(loss_f)
    final_acc_test.append(acc_f)
    
    
final_loss_2d=[]
final_acc_2d=[]

final_loss_val_2d=[]
final_acc_val_2d=[]

final_loss_test_2d=[]
final_acc_test_2d=[]

model_str = build_model(105)

res=[]
res.append(surface_2d_inp_C(init_layer[0], optim_layer[0]))
res.append(surface_2d_inp_b(init_layer[1], optim_layer[1]))
res.append(surface_2d_inp_C(init_layer[2], optim_layer[2]))
res.append(surface_2d_inp_b(init_layer[3], optim_layer[3]))
res.append(surface_2d_inp_X(init_layer[4], optim_layer[4]))
res.append(surface_2d_inp_b(init_layer[5], optim_layer[5]))
res.append(surface_2d_inp_X(init_layer[6], optim_layer[6]))
res.append(surface_2d_inp_b(init_layer[7], optim_layer[7]))
res.append(surface_2d_inp_X(init_layer[8], optim_layer[8]))
res.append(surface_2d_inp_b(init_layer[9], optim_layer[9]))


for k in range(81):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss_2d.append(loss_f)
    final_acc_2d.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val_2d.append(loss_f)
    final_acc_val_2d.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test_2d.append(loss_f)
    final_acc_test_2d.append(acc_f)
    
np.savetxt("GD_Mnist_lossepoch.csv", lossepoch, delimiter=",", fmt='%s')
np.savetxt("GD_Mnist_loss_tra.csv", lossx, delimiter=",", fmt='%s')
np.savetxt("GD_Mnist_acc_tra.csv", acctra, delimiter=",", fmt='%s')
np.savetxt("GD_Mnist_loss_test.csv", loss_test, delimiter=",", fmt='%s')
np.savetxt("GD_Mnist_acc_test.csv", acc_test, delimiter=",", fmt='%s')
np.savetxt("GD_Mnist_loss_val.csv", loss_val, delimiter=",", fmt='%s')
np.savetxt("GD_Mnist_acc_val.csv", acc_val, delimiter=",", fmt='%s')

#np.savetxt("xCR2_EVGO3_Mnist_lossepoch_test.csv", lossepoch_test, delimiter=",", fmt='%s')
#np.savetxt("xCR2_EVGO3_Mnist_skip.csv", skip, delimiter=",", fmt='%s')
np.savetxt("GD_MNIST_final_loss.csv", final_loss, delimiter=",", fmt='%s')
np.savetxt("GD_MNIST_final_acc.csv", final_acc, delimiter=",", fmt='%s')
np.savetxt("GD_MNIST_final_loss_val.csv", final_loss_val, delimiter=",", fmt='%s')
np.savetxt("GD_MNIST_final_acc_val.csv", final_loss_test, delimiter=",", fmt='%s')
np.savetxt("GD_MNIST_final_loss_test.csv", final_acc_val, delimiter=",", fmt='%s')
np.savetxt("GD_MNIST_final_acc_test.csv", final_acc_test, delimiter=",", fmt='%s')

np.savetxt("2d_GD_MNIST_final_loss.csv", final_loss_2d, delimiter=",", fmt='%s')
np.savetxt("2d_GD_MNIST_final_acc.csv", final_acc_2d, delimiter=",", fmt='%s')
np.savetxt("2d_GD_MNIST_final_loss_val.csv", final_loss_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_GD_MNIST_final_acc_val.csv", final_loss_test_2d, delimiter=",", fmt='%s')
np.savetxt("2d_GD_MNIST_final_loss_test.csv", final_acc_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_GD_MNIST_final_acc_test.csv", final_acc_test_2d, delimiter=",", fmt='%s')


# In[6]:


K.clear_session()


# # Adam

# In[7]:


def build_model(setseed):
    """
    Builds test Keras model for LeNet MNIST
    :param loss (str): Type of loss - must be one of Keras accepted keras losses
    :return: Keras dense model of predefined structure
    """
    input = Input(shape=input_shape)
    conv1 = Conv2D(6, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(input)
    avg1 = AveragePooling2D()(conv1)
    conv2 = Conv2D(16, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(avg1)
    avg2 = AveragePooling2D()(conv2)
    flat= Flatten()(avg2)
    dens1=Dense(units=120, activation='relu')(flat)
    dens2=Dense(units=84, activation='relu')(dens1)
    probs=Dense(num_classes, activation='softmax')(dens2)
    
    model = Model(input=input, output=probs)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

all_model = [None,None,None]
losses = [None,None,None]

prediction=[]

all_score =[0,0,0]
gr=[]
wr=[]
xwr=[]

for i in range(3):
    np.random.seed(25+i)
    model = build_model(i+2)
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
    
model.summary()


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




adamb_m = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_best')) for t in weig_best]
adamb_v = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='v_best')) for t in weig_best]
adam0_m = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_0')) for t in weig_non0]
adam0_v = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='v_0')) for t in weig_non0]
adam1_m = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_1')) for t in weig_non1]
adam1_v = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='v_2')) for t in weig_non1]

beta_1 = 0.9
beta_2 = 0.999
step_size = 0.001
eps = 1e-8
t = tf.Variable(1.0, name='iteration')

upd2=[]

for m, v, best, gbest,  param_i, in zip(adamb_m, adamb_v, weig_best, grad_best, wr[2]):
    _m = beta_1 * m + (1 - beta_1) * gbest
    _v = beta_2 * v + (1 - beta_2) * tf.pow(gbest, 2)
    m_hat = _m / (1 - tf.pow(beta_1, t))
    v_hat = _v / (1 - tf.pow(beta_2, t))
    #m_hat = tf.cast(m_hat, tf.float32)
    #v_hat = tf.cast(v_hat, tf.float32)
    #upd2.extend([(m, _m)])
    #upd2.extend([(v, _v)])
    upd2.extend([tf.assign(m, _m)])
    upd2.extend([tf.assign(v, _v)])
    xbest = best - step_size * m_hat / (tf.sqrt(v_hat) + eps)
    #upd2.extend([(param_i, xbest)])
    upd2.extend([tf.assign(param_i, xbest)])


upd2.extend([t.assign_add(1.0)])

upd2.extend([tf.assign(param_i, v)
        for param_i, v in zip(xwr[2], xweig_best)]
    )    


upd_bb2= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2],
                                                   minlos, prediction[0], prediction[1], prediction[2],
                                                  wr[2][0],wr[2][1],wr[2][2],wr[2][3],wr[2][4],
                                                  wr[2][5],wr[2][6],wr[2][7],wr[2][8],wr[2][9]], updates=upd2)



epochs=100 # degistir

lossepoch=[]
lossepoch_test=[]
lossx=[]
acctra=[]
loss_test=[]
acc_test=[]
skip=[]


loss_val=[]
acc_val=[]
lossepoch_val=[]


optim_layer=[]
init_layer=[]

for f in tqdm(range(epochs)):
    program_starts = time.time()
    tr1=[]
    tr2=[]
    res1=[]
    res2=[]
    res3=[]
    res4=[]
    print('Epoch', f)
    print ('train')
    
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    
    batches = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_train, y_train, batch_size=batch_size):
        K.set_learning_phase(1)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in Train mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_bb2(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:4])
        lossepoch.append(ll[2])
        tr1.append(ll[2])
        tr2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        skip.append(ll[3])
        batches += 1
        if batches > len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    m=(len(x_train) / batch_size)-int((len(x_train) / batch_size))
    tr1[-1]*=m
    tr2[-1]*=m
    lossx.append(np.mean(tr1))
    acctra.append(np.mean(tr2))
    print ('train loss score is :'+str(np.mean(tr1)))
    print ('train acc score is :'+str(np.mean(tr2)))
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))

    if f==epochs-1:
        optim_layer.append(ll[7])
        optim_layer.append(ll[8])
        optim_layer.append(ll[9])
        optim_layer.append(ll[10])
        optim_layer.append(ll[11])
        optim_layer.append(ll[12])
        optim_layer.append(ll[13])
        optim_layer.append(ll[14])
        optim_layer.append(ll[15])
        optim_layer.append(ll[16])
        print ("optim")  
    elif f==0:
        init_layer.append(ll[7])
        init_layer.append(ll[8])
        init_layer.append(ll[9])
        init_layer.append(ll[10])
        init_layer.append(ll[11])
        init_layer.append(ll[12])
        init_layer.append(ll[13])
        init_layer.append(ll[14])
        init_layer.append(ll[15])
        init_layer.append(ll[16])
        print ("init_layer")
    else:
        pass

    print ('test')
    batchesx = 0
    
    print ('validation')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in VAl mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_val.append(ll[2])
        res3.append(ll[2])
        res4.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_val) / batch_size:
            break
    m=(len(x_val) / batch_size)-int((len(x_val) / batch_size))
    res3[-1]*=m
    res4[-1]*=m
    loss_val.append(np.mean(res3))
    acc_val.append(np.mean(res4))
    print ('val loss score is :'+str(np.mean(res3)))
    print ('val acc score is :'+str(np.mean(res4)))
    print ('test')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_test, y_test, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in TEST mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_test.append(ll[2])
        res1.append(ll[2])
        res2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_test) / batch_size:
            break
    m=(len(x_test) / batch_size)-int((len(x_test) / batch_size))
    res1[-1]*=m
    res2[-1]*=m
    loss_test.append(np.mean(res1))
    acc_test.append(np.mean(res2))
    print ('test loss score is :'+str(np.mean(res1)))
    print ('test acc score is :'+str(np.mean(res2)))


    
print (min(lossx), np.argmin(lossx))
print (max(acctra), np.argmax(acctra))

print (min(loss_val), np.argmin(loss_val))
print (max(acc_val), np.argmax(acc_val))

print (min(loss_test), np.argmin(loss_test))
print (max(acc_test), np.argmax(acc_test))

print (acc_test[np.argmax(acc_val)], loss_test[np.argmax(acc_val)])
print (acc_test[np.argmin(loss_val)], loss_test[np.argmin(loss_val)])


res=[]

for j in range(10):
    res.append(surface_1d_inp(init_layer[j], optim_layer[j]))
    

final_loss=[]
final_acc=[]

final_loss_val=[]
final_acc_val=[]

final_loss_test=[]
final_acc_test=[]

model_str = build_model(105)

for k in range(50):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss.append(loss_f)
    final_acc.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val.append(loss_f)
    final_acc_val.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test.append(loss_f)
    final_acc_test.append(acc_f)
    
    
final_loss_2d=[]
final_acc_2d=[]

final_loss_val_2d=[]
final_acc_val_2d=[]

final_loss_test_2d=[]
final_acc_test_2d=[]

model_str = build_model(105)

res=[]
res.append(surface_2d_inp_C(init_layer[0], optim_layer[0]))
res.append(surface_2d_inp_b(init_layer[1], optim_layer[1]))
res.append(surface_2d_inp_C(init_layer[2], optim_layer[2]))
res.append(surface_2d_inp_b(init_layer[3], optim_layer[3]))
res.append(surface_2d_inp_X(init_layer[4], optim_layer[4]))
res.append(surface_2d_inp_b(init_layer[5], optim_layer[5]))
res.append(surface_2d_inp_X(init_layer[6], optim_layer[6]))
res.append(surface_2d_inp_b(init_layer[7], optim_layer[7]))
res.append(surface_2d_inp_X(init_layer[8], optim_layer[8]))
res.append(surface_2d_inp_b(init_layer[9], optim_layer[9]))


for k in range(81):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss_2d.append(loss_f)
    final_acc_2d.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val_2d.append(loss_f)
    final_acc_val_2d.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test_2d.append(loss_f)
    final_acc_test_2d.append(acc_f)
    
np.savetxt("Adam_Mnist_lossepoch.csv", lossepoch, delimiter=",", fmt='%s')
np.savetxt("Adam_Mnist_loss_tra.csv", lossx, delimiter=",", fmt='%s')
np.savetxt("Adam_Mnist_acc_tra.csv", acctra, delimiter=",", fmt='%s')
np.savetxt("Adam_Mnist_loss_test.csv", loss_test, delimiter=",", fmt='%s')
np.savetxt("Adam_Mnist_acc_test.csv", acc_test, delimiter=",", fmt='%s')
np.savetxt("Adam_Mnist_loss_val.csv", loss_val, delimiter=",", fmt='%s')
np.savetxt("Adam_Mnist_acc_val.csv", acc_val, delimiter=",", fmt='%s')

#np.savetxt("xCR2_Adam3_Mnist_lossepoch_test.csv", lossepoch_test, delimiter=",", fmt='%s')
#np.savetxt("xCR2_Adam3_Mnist_skip.csv", skip, delimiter=",", fmt='%s')
np.savetxt("Adam_MNIST_final_loss.csv", final_loss, delimiter=",", fmt='%s')
np.savetxt("Adam_MNIST_final_acc.csv", final_acc, delimiter=",", fmt='%s')
np.savetxt("Adam_MNIST_final_loss_val.csv", final_loss_val, delimiter=",", fmt='%s')
np.savetxt("Adam_MNIST_final_acc_val.csv", final_loss_test, delimiter=",", fmt='%s')
np.savetxt("Adam_MNIST_final_loss_test.csv", final_acc_val, delimiter=",", fmt='%s')
np.savetxt("Adam_MNIST_final_acc_test.csv", final_acc_test, delimiter=",", fmt='%s')

np.savetxt("2d_ADAM_MNIST_final_loss.csv", final_loss_2d, delimiter=",", fmt='%s')
np.savetxt("2d_ADAM_MNIST_final_acc.csv", final_acc_2d, delimiter=",", fmt='%s')
np.savetxt("2d_ADAM_MNIST_final_loss_val.csv", final_loss_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_ADAM_MNIST_final_acc_val.csv", final_loss_test_2d, delimiter=",", fmt='%s')
np.savetxt("2d_ADAM_MNIST_final_loss_test.csv", final_acc_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_ADAM_MNIST_final_acc_test.csv", final_acc_test_2d, delimiter=",", fmt='%s')


# In[8]:


K.clear_session()


# # CM

# In[9]:


def build_model(setseed):
    """
    Builds test Keras model for LeNet MNIST
    :param loss (str): Type of loss - must be one of Keras accepted keras losses
    :return: Keras dense model of predefined structure
    """
    input = Input(shape=input_shape)
    conv1 = Conv2D(6, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(input)
    avg1 = AveragePooling2D()(conv1)
    conv2 = Conv2D(16, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(avg1)
    avg2 = AveragePooling2D()(conv2)
    flat= Flatten()(avg2)
    dens1=Dense(units=120, activation='relu')(flat)
    dens2=Dense(units=84, activation='relu')(dens1)
    probs=Dense(num_classes, activation='softmax')(dens2)
    
    model = Model(input=input, output=probs)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

all_model = [None,None,None]
losses = [None,None,None]

prediction=[]

all_score =[0,0,0]
gr=[]
wr=[]
xwr=[]

for i in range(3):
    np.random.seed(25+i)
    model = build_model(i+2)
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
    
model.summary()


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

vb = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_best')) for t in weig_best]
vn0 = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_0')) for t in weig_non0]
vn1 = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_1')) for t in weig_non1]


step_size = 0.01
bet = 0.9

upd2=[]

for vx, best, gbest,  param_i, in zip(vb, weig_best, grad_best, wr[2]):
    _v = vx * bet
    _v = _v - step_size * gbest
    #_v = tf.cast(_v, tf.float32)
    upd2.extend([tf.assign(param_i, param_i + _v)])
    upd2.extend([tf.assign(vx, _v)])
    


upd2.extend([tf.assign(param_i, v)
        for param_i, v in zip(xwr[2], xweig_best)]
    )       


upd_bb2= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2],
                                                   minlos, prediction[0], prediction[1], prediction[2],
                                                  wr[2][0],wr[2][1],wr[2][2],wr[2][3],wr[2][4],
                                                  wr[2][5],wr[2][6],wr[2][7],wr[2][8],wr[2][9]], updates=upd2)



epochs=100 # degistir

lossepoch=[]
lossepoch_test=[]
lossx=[]
acctra=[]
loss_test=[]
acc_test=[]
skip=[]


loss_val=[]
acc_val=[]
lossepoch_val=[]


optim_layer=[]
init_layer=[]

for f in tqdm(range(epochs)):
    program_starts = time.time()
    tr1=[]
    tr2=[]
    res1=[]
    res2=[]
    res3=[]
    res4=[]
    print('Epoch', f)
    print ('train')
    
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    
    batches = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_train, y_train, batch_size=batch_size):
        K.set_learning_phase(1)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in Train mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_bb2(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:4])
        lossepoch.append(ll[2])
        tr1.append(ll[2])
        tr2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        skip.append(ll[3])
        batches += 1
        if batches > len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    m=(len(x_train) / batch_size)-int((len(x_train) / batch_size))
    tr1[-1]*=m
    tr2[-1]*=m
    lossx.append(np.mean(tr1))
    acctra.append(np.mean(tr2))
    print ('train loss score is :'+str(np.mean(tr1)))
    print ('train acc score is :'+str(np.mean(tr2)))
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))

    if f==epochs-1:
        optim_layer.append(ll[7])
        optim_layer.append(ll[8])
        optim_layer.append(ll[9])
        optim_layer.append(ll[10])
        optim_layer.append(ll[11])
        optim_layer.append(ll[12])
        optim_layer.append(ll[13])
        optim_layer.append(ll[14])
        optim_layer.append(ll[15])
        optim_layer.append(ll[16])
        print ("optim")  
    elif f==0:
        init_layer.append(ll[7])
        init_layer.append(ll[8])
        init_layer.append(ll[9])
        init_layer.append(ll[10])
        init_layer.append(ll[11])
        init_layer.append(ll[12])
        init_layer.append(ll[13])
        init_layer.append(ll[14])
        init_layer.append(ll[15])
        init_layer.append(ll[16])
        print ("init_layer")
    else:
        pass

    print ('test')
    batchesx = 0
    
    print ('validation')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in VAl mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_val.append(ll[2])
        res3.append(ll[2])
        res4.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_val) / batch_size:
            break
    m=(len(x_val) / batch_size)-int((len(x_val) / batch_size))
    res3[-1]*=m
    res4[-1]*=m
    loss_val.append(np.mean(res3))
    acc_val.append(np.mean(res4))
    print ('val loss score is :'+str(np.mean(res3)))
    print ('val acc score is :'+str(np.mean(res4)))
    print ('test')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_test, y_test, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in TEST mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_test.append(ll[2])
        res1.append(ll[2])
        res2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_test) / batch_size:
            break
    m=(len(x_test) / batch_size)-int((len(x_test) / batch_size))
    res1[-1]*=m
    res2[-1]*=m
    loss_test.append(np.mean(res1))
    acc_test.append(np.mean(res2))
    print ('test loss score is :'+str(np.mean(res1)))
    print ('test acc score is :'+str(np.mean(res2)))


    
print (min(lossx), np.argmin(lossx))
print (max(acctra), np.argmax(acctra))

print (min(loss_val), np.argmin(loss_val))
print (max(acc_val), np.argmax(acc_val))

print (min(loss_test), np.argmin(loss_test))
print (max(acc_test), np.argmax(acc_test))

print (acc_test[np.argmax(acc_val)], loss_test[np.argmax(acc_val)])
print (acc_test[np.argmin(loss_val)], loss_test[np.argmin(loss_val)])


res=[]

for j in range(10):
    res.append(surface_1d_inp(init_layer[j], optim_layer[j]))
    

final_loss=[]
final_acc=[]

final_loss_val=[]
final_acc_val=[]

final_loss_test=[]
final_acc_test=[]

model_str = build_model(105)

for k in range(50):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss.append(loss_f)
    final_acc.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val.append(loss_f)
    final_acc_val.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test.append(loss_f)
    final_acc_test.append(acc_f)
    
    
final_loss_2d=[]
final_acc_2d=[]

final_loss_val_2d=[]
final_acc_val_2d=[]

final_loss_test_2d=[]
final_acc_test_2d=[]

model_str = build_model(105)

res=[]
res.append(surface_2d_inp_C(init_layer[0], optim_layer[0]))
res.append(surface_2d_inp_b(init_layer[1], optim_layer[1]))
res.append(surface_2d_inp_C(init_layer[2], optim_layer[2]))
res.append(surface_2d_inp_b(init_layer[3], optim_layer[3]))
res.append(surface_2d_inp_X(init_layer[4], optim_layer[4]))
res.append(surface_2d_inp_b(init_layer[5], optim_layer[5]))
res.append(surface_2d_inp_X(init_layer[6], optim_layer[6]))
res.append(surface_2d_inp_b(init_layer[7], optim_layer[7]))
res.append(surface_2d_inp_X(init_layer[8], optim_layer[8]))
res.append(surface_2d_inp_b(init_layer[9], optim_layer[9]))


for k in range(81):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss_2d.append(loss_f)
    final_acc_2d.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val_2d.append(loss_f)
    final_acc_val_2d.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test_2d.append(loss_f)
    final_acc_test_2d.append(acc_f)
    
np.savetxt("CM_Mnist_lossepoch.csv", lossepoch, delimiter=",", fmt='%s')
np.savetxt("CM_Mnist_loss_tra.csv", lossx, delimiter=",", fmt='%s')
np.savetxt("CM_Mnist_acc_tra.csv", acctra, delimiter=",", fmt='%s')
np.savetxt("CM_Mnist_loss_test.csv", loss_test, delimiter=",", fmt='%s')
np.savetxt("CM_Mnist_acc_test.csv", acc_test, delimiter=",", fmt='%s')
np.savetxt("CM_Mnist_loss_val.csv", loss_val, delimiter=",", fmt='%s')
np.savetxt("CM_Mnist_acc_val.csv", acc_val, delimiter=",", fmt='%s')

#np.savetxt("xCR2_CM3_Mnist_lossepoch_test.csv", lossepoch_test, delimiter=",", fmt='%s')
#np.savetxt("xCR2_CM3_Mnist_skip.csv", skip, delimiter=",", fmt='%s')
np.savetxt("CM_MNIST_final_loss.csv", final_loss, delimiter=",", fmt='%s')
np.savetxt("CM_MNIST_final_acc.csv", final_acc, delimiter=",", fmt='%s')
np.savetxt("CM_MNIST_final_loss_val.csv", final_loss_val, delimiter=",", fmt='%s')
np.savetxt("CM_MNIST_final_acc_val.csv", final_loss_test, delimiter=",", fmt='%s')
np.savetxt("CM_MNIST_final_loss_test.csv", final_acc_val, delimiter=",", fmt='%s')
np.savetxt("CM_MNIST_final_acc_test.csv", final_acc_test, delimiter=",", fmt='%s')

np.savetxt("2d_CM_MNIST_final_loss.csv", final_loss_2d, delimiter=",", fmt='%s')
np.savetxt("2d_CM_MNIST_final_acc.csv", final_acc_2d, delimiter=",", fmt='%s')
np.savetxt("2d_CM_MNIST_final_loss_val.csv", final_loss_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_CM_MNIST_final_acc_val.csv", final_loss_test_2d, delimiter=",", fmt='%s')
np.savetxt("2d_CM_MNIST_final_loss_test.csv", final_acc_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_CM_MNIST_final_acc_test.csv", final_acc_test_2d, delimiter=",", fmt='%s')


# In[10]:


K.clear_session()


# # RmsProp

# In[11]:


def build_model(setseed):
    """
    Builds test Keras model for LeNet MNIST
    :param loss (str): Type of loss - must be one of Keras accepted keras losses
    :return: Keras dense model of predefined structure
    """
    input = Input(shape=input_shape)
    conv1 = Conv2D(6, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(input)
    avg1 = AveragePooling2D()(conv1)
    conv2 = Conv2D(16, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(avg1)
    avg2 = AveragePooling2D()(conv2)
    flat= Flatten()(avg2)
    dens1=Dense(units=120, activation='relu')(flat)
    dens2=Dense(units=84, activation='relu')(dens1)
    probs=Dense(num_classes, activation='softmax')(dens2)
    
    model = Model(input=input, output=probs)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

all_model = [None,None,None]
losses = [None,None,None]

prediction=[]

all_score =[0,0,0]
gr=[]
wr=[]
xwr=[]

for i in range(3):
    np.random.seed(25+i)
    model = build_model(i+2)
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
    
model.summary()


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




rmsb = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_best')) for t in weig_best]
rmsn0 = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_0')) for t in weig_non0]
rmsn1 = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_1')) for t in weig_non1]

step_size = 0.01
eps = 1e-6
bet = 0.9

upd2=[]

for rms, best, gbest,  param_i, in zip(rmsb, weig_best, grad_best, wr[2]):
    _rms = rms * bet
    _rms += (1 - bet) * gbest * gbest
    #_rms = tf.cast(_rms, tf.float32)
    rms_up =  -step_size * gbest / tf.sqrt(_rms + eps)

    upd2.extend([tf.assign(rms, _rms)])
    upd2.extend([tf.assign(param_i, best + rms_up)])


upd2.extend([tf.assign(param_i, v)
        for param_i, v in zip(xwr[2], xweig_best)]
    )    

upd_bb2= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2],
                                                   minlos, prediction[0], prediction[1], prediction[2],
                                                  wr[2][0],wr[2][1],wr[2][2],wr[2][3],wr[2][4],
                                                  wr[2][5],wr[2][6],wr[2][7],wr[2][8],wr[2][9]], updates=upd2)



epochs=100 # degistir

lossepoch=[]
lossepoch_test=[]
lossx=[]
acctra=[]
loss_test=[]
acc_test=[]
skip=[]


loss_val=[]
acc_val=[]
lossepoch_val=[]


optim_layer=[]
init_layer=[]

for f in tqdm(range(epochs)):
    program_starts = time.time()
    tr1=[]
    tr2=[]
    res1=[]
    res2=[]
    res3=[]
    res4=[]
    print('Epoch', f)
    print ('train')
    
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    
    batches = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_train, y_train, batch_size=batch_size):
        K.set_learning_phase(1)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in Train mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_bb2(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:4])
        lossepoch.append(ll[2])
        tr1.append(ll[2])
        tr2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        skip.append(ll[3])
        batches += 1
        if batches > len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    m=(len(x_train) / batch_size)-int((len(x_train) / batch_size))
    tr1[-1]*=m
    tr2[-1]*=m
    lossx.append(np.mean(tr1))
    acctra.append(np.mean(tr2))
    print ('train loss score is :'+str(np.mean(tr1)))
    print ('train acc score is :'+str(np.mean(tr2)))
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))

    if f==epochs-1:
        optim_layer.append(ll[7])
        optim_layer.append(ll[8])
        optim_layer.append(ll[9])
        optim_layer.append(ll[10])
        optim_layer.append(ll[11])
        optim_layer.append(ll[12])
        optim_layer.append(ll[13])
        optim_layer.append(ll[14])
        optim_layer.append(ll[15])
        optim_layer.append(ll[16])
        print ("optim")  
    elif f==0:
        init_layer.append(ll[7])
        init_layer.append(ll[8])
        init_layer.append(ll[9])
        init_layer.append(ll[10])
        init_layer.append(ll[11])
        init_layer.append(ll[12])
        init_layer.append(ll[13])
        init_layer.append(ll[14])
        init_layer.append(ll[15])
        init_layer.append(ll[16])
        print ("init_layer")
    else:
        pass

    print ('test')
    batchesx = 0
    
    print ('validation')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in VAl mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_val.append(ll[2])
        res3.append(ll[2])
        res4.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_val) / batch_size:
            break
    m=(len(x_val) / batch_size)-int((len(x_val) / batch_size))
    res3[-1]*=m
    res4[-1]*=m
    loss_val.append(np.mean(res3))
    acc_val.append(np.mean(res4))
    print ('val loss score is :'+str(np.mean(res3)))
    print ('val acc score is :'+str(np.mean(res4)))
    print ('test')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_test, y_test, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in TEST mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_test.append(ll[2])
        res1.append(ll[2])
        res2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_test) / batch_size:
            break
    m=(len(x_test) / batch_size)-int((len(x_test) / batch_size))
    res1[-1]*=m
    res2[-1]*=m
    loss_test.append(np.mean(res1))
    acc_test.append(np.mean(res2))
    print ('test loss score is :'+str(np.mean(res1)))
    print ('test acc score is :'+str(np.mean(res2)))


    
print (min(lossx), np.argmin(lossx))
print (max(acctra), np.argmax(acctra))

print (min(loss_val), np.argmin(loss_val))
print (max(acc_val), np.argmax(acc_val))

print (min(loss_test), np.argmin(loss_test))
print (max(acc_test), np.argmax(acc_test))

print (acc_test[np.argmax(acc_val)], loss_test[np.argmax(acc_val)])
print (acc_test[np.argmin(loss_val)], loss_test[np.argmin(loss_val)])


res=[]

for j in range(10):
    res.append(surface_1d_inp(init_layer[j], optim_layer[j]))
    

final_loss=[]
final_acc=[]

final_loss_val=[]
final_acc_val=[]

final_loss_test=[]
final_acc_test=[]

model_str = build_model(105)

for k in range(50):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss.append(loss_f)
    final_acc.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val.append(loss_f)
    final_acc_val.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test.append(loss_f)
    final_acc_test.append(acc_f)
    
    
final_loss_2d=[]
final_acc_2d=[]

final_loss_val_2d=[]
final_acc_val_2d=[]

final_loss_test_2d=[]
final_acc_test_2d=[]

model_str = build_model(105)

res=[]
res.append(surface_2d_inp_C(init_layer[0], optim_layer[0]))
res.append(surface_2d_inp_b(init_layer[1], optim_layer[1]))
res.append(surface_2d_inp_C(init_layer[2], optim_layer[2]))
res.append(surface_2d_inp_b(init_layer[3], optim_layer[3]))
res.append(surface_2d_inp_X(init_layer[4], optim_layer[4]))
res.append(surface_2d_inp_b(init_layer[5], optim_layer[5]))
res.append(surface_2d_inp_X(init_layer[6], optim_layer[6]))
res.append(surface_2d_inp_b(init_layer[7], optim_layer[7]))
res.append(surface_2d_inp_X(init_layer[8], optim_layer[8]))
res.append(surface_2d_inp_b(init_layer[9], optim_layer[9]))


for k in range(81):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss_2d.append(loss_f)
    final_acc_2d.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val_2d.append(loss_f)
    final_acc_val_2d.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test_2d.append(loss_f)
    final_acc_test_2d.append(acc_f)
    
np.savetxt("RmsProp_Mnist_lossepoch.csv", lossepoch, delimiter=",", fmt='%s')
np.savetxt("RmsProp_Mnist_loss_tra.csv", lossx, delimiter=",", fmt='%s')
np.savetxt("RmsProp_Mnist_acc_tra.csv", acctra, delimiter=",", fmt='%s')
np.savetxt("RmsProp_Mnist_loss_test.csv", loss_test, delimiter=",", fmt='%s')
np.savetxt("RmsProp_Mnist_acc_test.csv", acc_test, delimiter=",", fmt='%s')
np.savetxt("RmsProp_Mnist_loss_val.csv", loss_val, delimiter=",", fmt='%s')
np.savetxt("RmsProp_Mnist_acc_val.csv", acc_val, delimiter=",", fmt='%s')

#np.savetxt("xCR2_RmsProp3_Mnist_lossepoch_test.csv", lossepoch_test, delimiter=",", fmt='%s')
#np.savetxt("xCR2_RmsProp3_Mnist_skip.csv", skip, delimiter=",", fmt='%s')
np.savetxt("RmsProp_MNIST_final_loss.csv", final_loss, delimiter=",", fmt='%s')
np.savetxt("RmsProp_MNIST_final_acc.csv", final_acc, delimiter=",", fmt='%s')
np.savetxt("RmsProp_MNIST_final_loss_val.csv", final_loss_val, delimiter=",", fmt='%s')
np.savetxt("RmsProp_MNIST_final_acc_val.csv", final_loss_test, delimiter=",", fmt='%s')
np.savetxt("RmsProp_MNIST_final_loss_test.csv", final_acc_val, delimiter=",", fmt='%s')
np.savetxt("RmsProp_MNIST_final_acc_test.csv", final_acc_test, delimiter=",", fmt='%s')

np.savetxt("2d_RmsProp_MNIST_final_loss.csv", final_loss_2d, delimiter=",", fmt='%s')
np.savetxt("2d_RmsProp_MNIST_final_acc.csv", final_acc_2d, delimiter=",", fmt='%s')
np.savetxt("2d_RmsProp_MNIST_final_loss_val.csv", final_loss_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_RmsProp_MNIST_final_acc_val.csv", final_loss_test_2d, delimiter=",", fmt='%s')
np.savetxt("2d_RmsProp_MNIST_final_loss_test.csv", final_acc_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_RmsProp_MNIST_final_acc_test.csv", final_acc_test_2d, delimiter=",", fmt='%s')


# In[12]:


K.clear_session()


# # Adagrad

# In[13]:


def build_model(setseed):
    """
    Builds test Keras model for LeNet MNIST
    :param loss (str): Type of loss - must be one of Keras accepted keras losses
    :return: Keras dense model of predefined structure
    """
    input = Input(shape=input_shape)
    conv1 = Conv2D(6, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(input)
    avg1 = AveragePooling2D()(conv1)
    conv2 = Conv2D(16, (3,3), activation='relu', kernel_initializer=initializers.lecun_uniform(seed = setseed))(avg1)
    avg2 = AveragePooling2D()(conv2)
    flat= Flatten()(avg2)
    dens1=Dense(units=120, activation='relu')(flat)
    dens2=Dense(units=84, activation='relu')(dens1)
    probs=Dense(num_classes, activation='softmax')(dens2)
    
    model = Model(input=input, output=probs)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

all_model = [None,None,None]
losses = [None,None,None]

prediction=[]

all_score =[0,0,0]
gr=[]
wr=[]
xwr=[]

for i in range(3):
    np.random.seed(25+i)
    model = build_model(i+2)
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
    
model.summary()


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




agradb = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_best')) for t in weig_best]
agradn0 = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_0')) for t in weig_non0]
agradn1 = [tf.Variable(tf.zeros(t.shape, dtype=tf.float32, name='m_1')) for t in weig_non1]

step_size = 0.01
eps = 0.000001

upd2=[]

for agrad, best, gbest,  param_i, in zip(agradb, weig_best, grad_best, wr[2]):
    accgrad = agrad + gbest * gbest
    dx = - (step_size / tf.sqrt(accgrad + eps)) * gbest
    # accgrad = tf.cast(accgrad, tf.float32)
    # dx = tf.cast(dx, tf.float32)    
    upd2.extend([tf.assign(agrad, accgrad)])
    upd2.extend([tf.assign(param_i, best + dx)])


upd2.extend([tf.assign(param_i, v)
        for param_i, v in zip(xwr[2], xweig_best)]
    )    


upd_bb2= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2],
                                                   minlos, prediction[0], prediction[1], prediction[2],
                                                  wr[2][0],wr[2][1],wr[2][2],wr[2][3],wr[2][4],
                                                  wr[2][5],wr[2][6],wr[2][7],wr[2][8],wr[2][9]], updates=upd2)



epochs=100 # degistir

lossepoch=[]
lossepoch_test=[]
lossx=[]
acctra=[]
loss_test=[]
acc_test=[]
skip=[]


loss_val=[]
acc_val=[]
lossepoch_val=[]


optim_layer=[]
init_layer=[]

for f in tqdm(range(epochs)):
    program_starts = time.time()
    tr1=[]
    tr2=[]
    res1=[]
    res2=[]
    res3=[]
    res4=[]
    print('Epoch', f)
    print ('train')
    
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    
    batches = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_train, y_train, batch_size=batch_size):
        K.set_learning_phase(1)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in Train mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_bb2(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:4])
        lossepoch.append(ll[2])
        tr1.append(ll[2])
        tr2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        skip.append(ll[3])
        batches += 1
        if batches > len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    m=(len(x_train) / batch_size)-int((len(x_train) / batch_size))
    tr1[-1]*=m
    tr2[-1]*=m
    lossx.append(np.mean(tr1))
    acctra.append(np.mean(tr2))
    print ('train loss score is :'+str(np.mean(tr1)))
    print ('train acc score is :'+str(np.mean(tr2)))
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))

    if f==epochs-1:
        optim_layer.append(ll[7])
        optim_layer.append(ll[8])
        optim_layer.append(ll[9])
        optim_layer.append(ll[10])
        optim_layer.append(ll[11])
        optim_layer.append(ll[12])
        optim_layer.append(ll[13])
        optim_layer.append(ll[14])
        optim_layer.append(ll[15])
        optim_layer.append(ll[16])
        print ("optim")  
    elif f==0:
        init_layer.append(ll[7])
        init_layer.append(ll[8])
        init_layer.append(ll[9])
        init_layer.append(ll[10])
        init_layer.append(ll[11])
        init_layer.append(ll[12])
        init_layer.append(ll[13])
        init_layer.append(ll[14])
        init_layer.append(ll[15])
        init_layer.append(ll[16])
        print ("init_layer")
    else:
        pass

    print ('test')
    batchesx = 0
    
    print ('validation')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in VAl mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_val.append(ll[2])
        res3.append(ll[2])
        res4.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_val) / batch_size:
            break
    m=(len(x_val) / batch_size)-int((len(x_val) / batch_size))
    res3[-1]*=m
    res4[-1]*=m
    loss_val.append(np.mean(res3))
    acc_val.append(np.mean(res4))
    print ('val loss score is :'+str(np.mean(res3)))
    print ('val acc score is :'+str(np.mean(res4)))
    print ('test')
    batchesx = 0
    for x_batch, y_batch in ImageDataGenerator().flow(x_test, y_test, batch_size=batch_size):
        K.set_learning_phase(0)
        inputs = [x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  1, # learning phase in TEST mode
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                  x_batch, # X
                  np.ones(y_batch.shape[0]), # sample weights
                  y_batch, # y
                 ]
        ll = upd_test(inputs)
        yhat=ll[6]
        #print (accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        #print (ll[:3])
        lossepoch_test.append(ll[2])
        res1.append(ll[2])
        res2.append(accuracy_score(np.argmax(y_batch,axis=1), np.argmax(yhat,axis=1)))
        batchesx += 1
        if batchesx >= len(x_test) / batch_size:
            break
    m=(len(x_test) / batch_size)-int((len(x_test) / batch_size))
    res1[-1]*=m
    res2[-1]*=m
    loss_test.append(np.mean(res1))
    acc_test.append(np.mean(res2))
    print ('test loss score is :'+str(np.mean(res1)))
    print ('test acc score is :'+str(np.mean(res2)))


    
print (min(lossx), np.argmin(lossx))
print (max(acctra), np.argmax(acctra))

print (min(loss_val), np.argmin(loss_val))
print (max(acc_val), np.argmax(acc_val))

print (min(loss_test), np.argmin(loss_test))
print (max(acc_test), np.argmax(acc_test))

print (acc_test[np.argmax(acc_val)], loss_test[np.argmax(acc_val)])
print (acc_test[np.argmin(loss_val)], loss_test[np.argmin(loss_val)])


res=[]

for j in range(10):
    res.append(surface_1d_inp(init_layer[j], optim_layer[j]))
    

final_loss=[]
final_acc=[]

final_loss_val=[]
final_acc_val=[]

final_loss_test=[]
final_acc_test=[]

model_str = build_model(105)

for k in range(50):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss.append(loss_f)
    final_acc.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val.append(loss_f)
    final_acc_val.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test.append(loss_f)
    final_acc_test.append(acc_f)
    
    
final_loss_2d=[]
final_acc_2d=[]

final_loss_val_2d=[]
final_acc_val_2d=[]

final_loss_test_2d=[]
final_acc_test_2d=[]

model_str = build_model(105)

res=[]
res.append(surface_2d_inp_C(init_layer[0], optim_layer[0]))
res.append(surface_2d_inp_b(init_layer[1], optim_layer[1]))
res.append(surface_2d_inp_C(init_layer[2], optim_layer[2]))
res.append(surface_2d_inp_b(init_layer[3], optim_layer[3]))
res.append(surface_2d_inp_X(init_layer[4], optim_layer[4]))
res.append(surface_2d_inp_b(init_layer[5], optim_layer[5]))
res.append(surface_2d_inp_X(init_layer[6], optim_layer[6]))
res.append(surface_2d_inp_b(init_layer[7], optim_layer[7]))
res.append(surface_2d_inp_X(init_layer[8], optim_layer[8]))
res.append(surface_2d_inp_b(init_layer[9], optim_layer[9]))


for k in range(81):
    model_str.layers[1].set_weights([res[0][k],res[1][k]])
    model_str.layers[3].set_weights([res[2][k],res[3][k]])
    model_str.layers[6].set_weights([res[4][k],res[5][k]])
    model_str.layers[7].set_weights([res[6][k],res[7][k]])
    model_str.layers[8].set_weights([res[8][k],res[9][k]])
    
    loss_f, acc_f = model_str.evaluate(x_train,y_train)

    final_loss_2d.append(loss_f)
    final_acc_2d.append(acc_f)
    
    
    loss_f, acc_f = model_str.evaluate(x_val,y_val)

    final_loss_val_2d.append(loss_f)
    final_acc_val_2d.append(acc_f)
    
    loss_f, acc_f = model_str.evaluate(x_test,y_test)

    final_loss_test_2d.append(loss_f)
    final_acc_test_2d.append(acc_f)
    
np.savetxt("Adagrad_Mnist_lossepoch.csv", lossepoch, delimiter=",", fmt='%s')
np.savetxt("Adagrad_Mnist_loss_tra.csv", lossx, delimiter=",", fmt='%s')
np.savetxt("Adagrad_Mnist_acc_tra.csv", acctra, delimiter=",", fmt='%s')
np.savetxt("Adagrad_Mnist_loss_test.csv", loss_test, delimiter=",", fmt='%s')
np.savetxt("Adagrad_Mnist_acc_test.csv", acc_test, delimiter=",", fmt='%s')
np.savetxt("Adagrad_Mnist_loss_val.csv", loss_val, delimiter=",", fmt='%s')
np.savetxt("Adagrad_Mnist_acc_val.csv", acc_val, delimiter=",", fmt='%s')

#np.savetxt("xCR2_Adagrad3_Mnist_lossepoch_test.csv", lossepoch_test, delimiter=",", fmt='%s')
#np.savetxt("xCR2_Adagrad3_Mnist_skip.csv", skip, delimiter=",", fmt='%s')
np.savetxt("Adagrad_MNIST_final_loss.csv", final_loss, delimiter=",", fmt='%s')
np.savetxt("Adagrad_MNIST_final_acc.csv", final_acc, delimiter=",", fmt='%s')
np.savetxt("Adagrad_MNIST_final_loss_val.csv", final_loss_val, delimiter=",", fmt='%s')
np.savetxt("Adagrad_MNIST_final_acc_val.csv", final_loss_test, delimiter=",", fmt='%s')
np.savetxt("Adagrad_MNIST_final_loss_test.csv", final_acc_val, delimiter=",", fmt='%s')
np.savetxt("Adagrad_MNIST_final_acc_test.csv", final_acc_test, delimiter=",", fmt='%s')

np.savetxt("2d_Adagrad_MNIST_final_loss.csv", final_loss_2d, delimiter=",", fmt='%s')
np.savetxt("2d_Adagrad_MNIST_final_acc.csv", final_acc_2d, delimiter=",", fmt='%s')
np.savetxt("2d_Adagrad_MNIST_final_loss_val.csv", final_loss_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_Adagrad_MNIST_final_acc_val.csv", final_loss_test_2d, delimiter=",", fmt='%s')
np.savetxt("2d_Adagrad_MNIST_final_loss_test.csv", final_acc_val_2d, delimiter=",", fmt='%s')
np.savetxt("2d_Adagrad_MNIST_final_acc_test.csv", final_acc_test_2d, delimiter=",", fmt='%s')


# In[ ]:




