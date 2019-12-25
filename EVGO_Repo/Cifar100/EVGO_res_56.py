#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division

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
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelBinarizer
from keras import initializers
import time
from tqdm import tqdm
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model

from keras.optimizers import SGD


# In[2]:


# Training parameters
batch_size = 128  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 100

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True


# In[3]:


n = 9

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[4]:


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 0.13
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=100):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy', metrics = ['accuracy'])
    return model


# In[5]:


depth


# In[6]:


from keras.layers import Input
import keras.backend as K
from keras.models import Model

all_model = [None,None,None]
losses = [None,None,None]

prediction=[]

all_score =[0,0,0]
gr=[]
wr=[]
xwr=[]

for i in range(3):
    np.random.seed(25+i)
    set_random_seed(25+i)
    model = resnet_v1(input_shape=input_shape, depth=depth)
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


# In[7]:


model.summary()


# In[8]:


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


# In[9]:


minlos = K.argmin(losses)

grr=[]
for x in gr:
    for y in x:
        grr.append(y)


# In[10]:


upd_test= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2], minlos, prediction[0], prediction[1], prediction[2] ])


# In[11]:


len(xwr[0])


# In[ ]:





# In[ ]:





# In[12]:


import tensorflow as tf


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
    
    nbb0 = gr_ck2[0:minlos]                       #[0,enk) U (enk,] araliklarinin birlesimi bize nonbesti verecek
    nbb1 = gr_ck2[minlos+1:]                      #[0,enk) U (enk,] araliklarinin birlesimi bize nonbesti verecek
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
    wnbb0 = wr_ck2[0:minlos]                       #[0,enk) U (enk,] araliklarinin birlesimi bize nonbesti verecek
    wnbb1 = wr_ck2[minlos+1:]                      #[0,enk) U (enk,] araliklarinin birlesimi bize nonbesti verecek
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
        xwnbb0 = xwr_ck2[0:minlos]                       #[0,enk) U (enk,] araliklarinin birlesimi bize nonbesti verecek
        xwnbb1 = xwr_ck2[minlos+1:]                      #[0,enk) U (enk,] araliklarinin birlesimi bize nonbesti verecek
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


# In[13]:


los=tf.stack([losses[0], losses[1], losses[2]])

newshape = (3, )
los2=tf.reshape(los, newshape) 
losbest = los2[minlos]

#wb = wr_ck[minlos]
los_0 = los2[0:minlos]                       #[0,enk) U (enk,] araliklarinin birlesimi bize nonbesti verecek
los_1 = los2[minlos+1:]                      #[0,enk) U (enk,] araliklarinin birlesimi bize nonbesti verecek
loswnbc = tf.concat([los_0,los_1],0)    
loswnbc = tf.reshape(loswnbc,(-1,))
newshape2 = (2, )

loswnbc2 = tf.reshape(loswnbc, newshape2)
losss0 = loswnbc2[0]
losss1 = loswnbc2[1]



# In[14]:

eps = 1.5


mn0 = [tf.keras.backend.l2_normalize((best-nonbest)*(losbest-losss0)/ tf.reduce_sum(tf.pow((best-nonbest),2)+eps))  for best, nonbest in zip(weig_best, weig_non0)]

mn1 = [tf.keras.backend.l2_normalize((best-nonbest)*(losbest-losss1)/ tf.reduce_sum(tf.pow((best-nonbest),2)+eps))  for best, nonbest in zip(weig_best, weig_non1)]



nCom0 = [non- model.optimizer.lr* grad - (model.optimizer.lr/10) * mn for mn, grad, non in zip(mn0,grad_non0, weig_non0 )]

nCom1 = [non- model.optimizer.lr* grad - (model.optimizer.lr/10) * mn for mn, grad, non in zip(mn1,grad_non1, weig_non1 )]

xbest = [ -model.optimizer.lr* nc + non for nc, non in zip(grad_best, weig_best)]



# In[15]:



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


upd_bb2= K.function(inputs=input_tensors, outputs=[ losses[0], losses[1], losses[2], minlos, prediction[0], prediction[1], prediction[2] ], updates=upd2)

datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

datagentest = ImageDataGenerator()

# In[ ]:





# In[17]:


# alfa 0.001 beta 0.001 for nonbest, 0.1 alfa for best
from sklearn.metrics import accuracy_score

lossepoch=[]
lossepoch_test=[]
lossx=[]
acctra=[]
loss_test=[]
acc_test=[]
skip=[]

for f in tqdm(range(200)):
    program_starts = time.time()
    K.set_value(all_model[0].optimizer.lr, lr_schedule(f))
    K.set_value(all_model[1].optimizer.lr, lr_schedule(f))
    K.set_value(all_model[2].optimizer.lr, lr_schedule(f))
    tr1=[]
    tr2=[]
    res1=[]
    res2=[]
    print('Epoch', f)
    print ('train')
    
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:

    datagen.fit(x_train)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
        K.set_learning_phase(1)
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
        ll = upd_bb2(inputs)
        yhat=ll[6]
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
    print ('test')
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))
    print ('test')
    batchesx = 0

    K.set_learning_phase(0)
    inputs = [x_test, # X
              np.ones(y_test.shape[0]), # sample weights
              y_test, # y
              1, # learning phase in TEST mode
              x_test, # X
              np.ones(y_test.shape[0]), # sample weights
              y_test, # y
              x_test, # X
              np.ones(y_test.shape[0]), # sample weights
              y_test, # y
             ]
    ll = upd_test(inputs)
    yhat=ll[6]
    lossepoch_test.append(ll[2])
    loss_test.append(ll[2])
    acc_test.append(accuracy_score(np.argmax(y_test,axis=1), np.argmax(yhat,axis=1)))
    print ('test loss score is :'+str(ll[2]))
    print ('test acc score is :'+str(accuracy_score(np.argmax(y_test,axis=1), np.argmax(yhat,axis=1))))


# In[18]:




np.savetxt("freya_alg4_lossepoch_r56_cifar100_reduce_sum.csv", lossepoch, delimiter=",", fmt='%s')
np.savetxt("freya_alg4_lossepoch_test_r56_cifar100_reduce_sum.csv", lossepoch_test, delimiter=",", fmt='%s')       
np.savetxt("freya_alg4_loss_tra_r56_cifar100_reduce_sum.csv", lossx, delimiter=",", fmt='%s')
np.savetxt("freya_alg4_skip_r56_cifar100_reduce_sum.csv", skip, delimiter=",", fmt='%s')
np.savetxt("freya_alg4_acc_tra_r56_cifar100_reduce_sum.csv", acctra, delimiter=",", fmt='%s')
np.savetxt("freya_alg4_loss_test_r56_cifar100_reduce_sum.csv", loss_test, delimiter=",", fmt='%s')
np.savetxt("freya_alg4_acc_test_r56_cifar100_reduce_sum.csv", acc_test, delimiter=",", fmt='%s')


print (np.max(acc_test), np.argmax(acc_test))
print(np.min(loss_test), np.argmin(loss_test))
print (np.min(lossx), np.argmin(lossx))
print (np.max(acctra), np.argmax(acctra))




