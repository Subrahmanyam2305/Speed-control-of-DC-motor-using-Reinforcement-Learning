#!/usr/bin/env python
# coding: utf-8

# ## Importing relavant libraries ##

# In[65]:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd                    ## To manipulate and analyse the data(Usually from a csv File)
import matplotlib.pyplot as plt        ## To visualise plots
import math                            ## To perform mathematical operations
from keras.models import Sequential    ## To load model sequential from keras library
import numpy as np                     ## To perform operations on arrays     
from keras.layers import Dense   ## To load layers Dense and LSTM

from sklearn.metrics import mean_squared_error   ## To load the modeule MSE from scikit
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
np.random.seed(1) # random seed 1 chosen after trial and error on best fit to experimental data 


# ## Data Preperation ##

# In[66]:


# load the DC_Motor dataset - SISO - single input single output 
dataset = pd.read_csv('DC_Motor1.csv', header = None)
print(dataset.head())
dataset = dataset.drop_duplicates(subset=[1],keep='last')
dataset = dataset.drop_duplicates(subset=[0],keep='last')
dataset = dataset.values
dataset.shape
print(dataset)
# Normalize the data using Minmax scaler to lie between 0 and 1
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)


# In[67]:


# Demarcating the data for training and testing
train_len = int(0.7*len(dataset))
test_len = len(dataset) - train_len
train_data = dataset[0:train_len,:]
test_data = dataset[train_len:len(dataset),:]

# Dividing to train input and output
X_train = train_data[:,0]
y_train = train_data[:,1]

# Dividing into test input and output
X_test = test_data[:,0]
y_test = test_data[:,1]


# In[68]:


print(X_train)
print(y_train)


# In[69]:


plt.figure(figsize=(20, 20))
plt.scatter(X_train,y_train)


# ## Initializing model ##

# In[70]:


# Feed Forward Neural Network with 1 fully connected hidden layers and 1 output layer
model = Sequential()
# Layer 1 with 40 neurons with activation function as Rectified Linear Unit
model.add(Dense(40, input_dim=1, activation='relu'))  
model.add(Dense(30))
model.add(Dense(1))


# ## Compiling model ##

# In[71]:


# Model compilation with a validation split of 33%; displaying the errors at each epoch
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train,validation_split=0.33, epochs=2000, batch_size=16)
#batch size 16 optimum among 8,16,32

"""
# ## Model performance ##

# In[72]:


# Plot training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('prediction')
plt.xlabel('epoch')
plt.legend(['loss', 'validation loss'], loc='upper right')
font = {'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
plt.grid()
plt.show()


# In[73]:


# Predicting train and test outputs using obtained model
trainPredict = model.predict(X_train)

testPredict = model.predict(X_test)


# In[74]:


# Calculating RMSE for normalized data
trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
print('Train error: %.10f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test, testPredict))
print('Test error: %.10f RMSE' % (testScore))


# In[75]:


trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
print('Train error: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test, testPredict))
print('Test error: %.2f RMSE' % (testScore))
print(testPredict.shape)
print(X_test.shape)
print(X_test)
print(y_test)
a=X_test
b=y_test
plt.plot(a,b,color='darkorange',lw=1)


# ## Visualization of predicted and experimental outputs on test data ##

# In[76]:


width = 12
height = 7

plt.figure(figsize=(width, height))
plt.plot(X_test, y_test, color='darkorange')
#plt.hold('on')
plt.plot(X_test,testPredict, color='navy')
plt.title('Experimental vs Predicted output')
plt.ylabel('Rotor speed ')
plt.xlabel('Number of samples')
#plt.xlim(0,255)
#plt.ylim(0,3800)
plt.legend(['train','train predict','testpredict'], loc='upper right')
font = {'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)
plt.grid()
plt.show()


# In[77]:


plt.close


# In[79]:


#model.predict([255])


# In[ ]:

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


"""

