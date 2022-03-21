#!/usr/bin/env python
# coding: utf-8

# # Hackathon : Neural network model applied for predictions

# During this work, the purpose is to make <ins>prediction</ins> on a test set. In order to do this, we will first apply a **Deep Learning** model based on a **Neural Network** method and train this model thanks to a train set

# **Librairies needed**

# In[1]:


#General librairies that we usually need
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

# Math librairies
import math
import random

# Data treatment librairies
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Model of neural network librairies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Improvement of our model librairies
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


# ## I) Dataset treatment

# ### Importation of the dataset

# My student number is **710053**. Indeed, I chose the dataset number **3** because these is no dataset **53**.

# I first treated the excel in order to have a first line that contains the titles of the columns and to not drop the first data during the importation

# In[2]:


data_train = pd.read_csv('phpYLeydd_train_3.csv',sep=',')
data_train


# In[3]:


data_train.info()


# In[4]:


data_train.shape


# We have 32 **features** <ins>(X)</ins> and 1 **layer** <ins>(y)</ins> to predict for a total of <ins>7869</ins> data.

# In[5]:


data_train['y'].unique()


# In[6]:


data_train['y'].nunique()


# So we have **5 labels**

# I decide to convert the labels into **numeric values**. It would be easier to work on it. Each number from 1 to 5 corresponds to a class. 

# In[7]:


data_train['y'] = data_train['y'].replace("'R'", 1)
data_train['y'] = data_train['y'].replace("'D'", 2)
data_train['y'] = data_train['y'].replace("'S'", 3)
data_train['y'] = data_train['y'].replace("'H'", 4)
data_train['y'] = data_train['y'].replace("'P'", 5)


# ### Visualization

# In[8]:


plt.scatter(
    data_train['x1'],
    data_train['x24'],
    c = data_train['y'], 
    cmap='viridis')
plt.xlabel('x1')
plt.ylabel('x24')
plt.title('Plot of the data')
plt.show()


# We plot the data according to 2 features <ins>x1</ins> and <ins>x24</ins>.

# ### Convert the dataset into a matrix

# We have imported the train dataset. We have now to treat it in order to work on it.
# First, we have to store our values in arrays. Matrix are much easier to use and simplify maths used for the model and is much more faster to compile.

# In[9]:


X_train = data_train.iloc[:,:data_train.shape[1]-1]
X_train = X_train.values
print('X :',X_train)


# In[10]:


Y_train = data_train.iloc[:,-1]
Y_train
print('Y :\n',Y_train)


# In[11]:


print('X_train shape :',X_train.shape)

print('Y_train shape :',Y_train.shape)


# We have now 2 arrays with the good shape:
# - **X** : features
# - **Y** : layer to predict

# ### Pre processing and scaling 

# In[12]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_train


# We use **MinMaxScaler** function that gets the max of the inputs to devide all the others by the max. This way, all the inputs are **scaled** (between 0 and 1)

# In[13]:


Y_train = to_categorical(Y_train)
Y_train = np.delete(Y_train,[0],1)
Y_train


# We use the **hot encoding method** for the label. It create a matrix with 5 columns (number of labels). If a data belongs to a label, the value corresponding to the index of will be <ins>1</ins> and the rest will be evaluated to <ins>0</ins>. For example, we can see that the first data belong to label 1 ('R'), the last one to label 3('S').

# ### Import and treat the test set

# In[14]:


data_test = pd.read_csv('phpYLeydd_test.csv',sep=',')
data_test['y'] = data_test['y'].replace("'R'", 1)
data_test['y'] = data_test['y'].replace("'D'", 2)
data_test['y'] = data_test['y'].replace("'S'", 3)
data_test['y'] = data_test['y'].replace("'H'", 4)
data_test['y'] = data_test['y'].replace("'P'", 5)
data_test


# In[15]:


X_test_val = data_test.iloc[:,:data_test.shape[1]-1]
X_test_val = X_test_val.values
print('X :',X_test_val)


# In[16]:


min_max_scaler = preprocessing.MinMaxScaler()
X_test_val = min_max_scaler.fit_transform(X_test_val)
X_test_val


# In[17]:


Y_test_val = data_test.iloc[:,-1]
Y_test_val
print('Y :\n',Y_test_val)


# In[18]:


Y_test_val = to_categorical(Y_test_val)
Y_test_val = np.delete(Y_test_val,[0],1)
Y_test_val


# In[19]:


print('X_test shape :',X_test_val.shape)

print('Y_test shape :',Y_test_val.shape)


# In[20]:


X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size=0.5)


# For the **validation** set, we split the test set into 2 parts <ins>(50/50)</ins>

# In[21]:


print('Train set:')
print('X_train shape:',X_train.shape)
print('Y_train shape:',Y_train.shape)

print('\nTest set:')
print('X_test shape:',X_test.shape)
print('Y_test shape:',Y_test.shape)

print('\nValidation set:')
print('X_val shape:',X_val.shape)
print('Y_val shape:',Y_val.shape)


# By printing the shapes, we could verify that:
# - <ins>50%</ins> (50% of test set) is for the **test set**
# - <ins>50%</ins> (50% of test set) is for the **validation set**

# ## II) Neural Network model

# We will build a **Deep Learning** model based on **Neural Network** methods. We barely have to:
# - **initialize the network** with random parameters thanks to a linear function **(Zi = Wi * Xi + Bi)**
# - apply an **activation function**
# - settle a **forward propagation**
# The steps consists in building our neural network from inputs to outputs going through hidden layers (nodes) interconnected. In order to improve the model by finding good parameters W and B, we have to:
# - calculate the **loss function** (cost function)
# - update the parameters with a **back propagation**
# 
# These are the fundamental steps of our model.
# We could import librairies that apply directly a model.

# ### Implementation of the model

# In[22]:


nb_neurons = 32

# Input layer : the number of features

model = Sequential([
    Dense(nb_neurons, activation='relu', input_shape=(X_train.shape[1],)), #Hidden layer 1
    Dense(nb_neurons, activation='relu'), # Hidden layer 2
    Dense(Y_train.shape[1], activation='softmax'), # Output layer
])


# In[23]:


model.summary()


# Sequential function creates a fully connected neural network composed of:
# - input layer
# - 2 hidden layers composed of 32 neurons
# - output layer
# For the activation, we chose the **ReLu function : 1/1+exp(-Zi)**

# In[24]:


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# To optimize our model, we use have to choose:
# - **optimizer**: we chose **SGD** that refers to **stochastic gradient descent (mini-batch gradient descent)**
# - **loss function** :calculate the error of our model with a <ins>categorical-crossentropy</ins>
# - we track the **accuracy** as metric

# In[25]:


n_epochs = 100
n_batch = 32

hist = model.fit(X_train, Y_train,
          batch_size=n_batch, epochs=n_epochs,
          validation_data=(X_val, Y_val))


# In[26]:


print('Model performance:')
print('\nModel error:',model.evaluate(X_test, Y_test)[0])
print('\nModel accuracy:',model.evaluate(X_test, Y_test)[1]*100,'%')


# ### Improvement of the model with regularization

# We will now try to better our model with a regularization method with the **Adam** optimizer.

# In[31]:


nb_neurons = 50

model_reg = Sequential([
    Dense(nb_neurons, activation='relu', input_shape=(X_train.shape[1],)), #Input
    Dense(nb_neurons, activation='relu'), #Hidden layer 1
    Dense(nb_neurons, activation='relu'), #Hidden layer 2
    Dense(nb_neurons, activation='relu'), #Hidden layer 3
    Dense(Y_train.shape[1], activation='softmax'),
])


# In[32]:


model_reg.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[33]:


n_batch = 32
n_epochs = 100

hist_reg = model_reg.fit(X_train, Y_train,
          batch_size=n_batch, epochs=n_epochs,
          validation_data=(X_val, Y_val))


# Changes are:
# - we added neurons and hidden layers because a more complex neural network is more accurate (regardless overfitting)
# - we changed the optimizer for the **Addam** one.

# In[34]:


print('Model performance:')
print('\nModel error:',model_reg.evaluate(X_test, Y_test)[0])
print('\nModel accuracy:',model_reg.evaluate(X_test, Y_test)[1]*100,'%')


# ### Improvement with regularization and drop out

# In[35]:


dropout_rate = 0.3

model_reg_drop = Sequential([
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(dropout_rate),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(dropout_rate),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(dropout_rate),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(dropout_rate),
    Dense(Y_train.shape[1], activation='softmax', kernel_regularizer=regularizers.l2(0.01)),
])


# In[36]:


model_reg_drop.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[37]:


n_batch = 32
n_epochs = 100

hist_reg_drop = model_reg_drop.fit(X_train, Y_train,
          batch_size=n_batch, epochs=n_epochs,
          validation_data=(X_val, Y_val))


# The changes are:
# - we added the **drop out** which means that the neurons in the previous layer has a probability of 0.3 in dropping out during training
# - with the **regularization L2**, the squared values of those parameters in our overall loss function, and weight them by 0.01 in the loss function

# In[38]:


print('Model performance:')
print('\nModel error:',model_reg_drop.evaluate(X_test, Y_test)[0])
print('\nModel accuracy:',model_reg_drop.evaluate(X_test, Y_test)[1]*100,'%')


# This model may have **overtifitting issues** because the accuracy is not improved passed few epochs.

# ### Plots

# **First model**

# In[39]:


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.title('Evolution of errors')

plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.title('Evolution of accuracy')


# With more and more iterations, our model shows us a deep decrease of its errors. The more we train our model, the more it is accurate until a stabilisation. Indeed, we do not have **overfitting problems**.

# **Model with regularization**

# In[40]:


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(hist_reg.history['loss'])
plt.plot(hist_reg.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.title('Evolution of errors with regularization')

plt.subplot(1,2,2)
plt.plot(hist_reg.history['accuracy'])
plt.plot(hist_reg.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.title('Evolution of accuracy with regularization')


# **Model with regularization L2 and drop out**

# In[41]:


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(hist_reg_drop.history['loss'])
plt.plot(hist_reg_drop.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.title('Evolution of errors with regularization and drop out')

plt.subplot(1,2,2)
plt.plot(hist_reg_drop.history['accuracy'])
plt.plot(hist_reg_drop.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.title('Evolution of accuracy with regularization and drop out')


# We could conclude that we can improve our model by chosing the regularization model by:
# - adding neurons
# - adding epochs

# ### Improvement of the model

# In[57]:


nb_neurons = 50

model_opti = Sequential([
    Dense(nb_neurons, activation='relu', input_shape=(X_train.shape[1],)), #Input layer
    Dense(nb_neurons, activation='relu'), #Hidden layer 1
    Dense(nb_neurons, activation='relu'), #Hidden layer 2
    Dense(nb_neurons, activation='relu'), #Hidden layer 3
    Dense(Y_train.shape[1], activation='softmax'), #Output layer
])


# In[58]:


model_opti.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[59]:


n_batch = 32
n_epochs = 200

hist_opti = model_opti.fit(X_train, Y_train,
          batch_size=n_batch, epochs=n_epochs,
          validation_data=(X_val, Y_val))


# In[60]:


print('Model performance:')
print('\nModel error:',model_opti.evaluate(X_test, Y_test)[0])
print('\nModel accuracy:',model_opti.evaluate(X_test, Y_test)[1]*100,'%')


# ### Plot of optimized model

# In[52]:


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(hist_opti.history['loss'])
plt.plot(hist_opti.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.title('Evolution of errors with optimized parameters')

plt.subplot(1,2,2)
plt.plot(hist_opti.history['accuracy'])
plt.plot(hist_opti.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.title('Evolution of accuracy with optimized parameters')


# ### Conclusion

# After this work, we obtained a model with:
# - **loss : 3.3**
# - **accuracy rate max obtained: 35%** with regularisation with neurons=50, batch=32 and epochs = 200
# 
# This is an acceptable accuracy but this model is disturbed by **overfitting issues**. 
