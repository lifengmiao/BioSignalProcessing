# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:26:00 2017

@author: lifeng.miao
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:55:47 2017

@author: lifeng.miao
"""
from datetime import datetime
import wfdb
import numpy as np
import scipy
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.utils import plot_model,to_categorical
from keras import optimizers
from keras import callbacks
from glob import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
ecg_N = []
#subjects = list(range(100,110))+list(range(111,120))+list(range(121,125))
#subjects = [101, 106, 108, 109, 112, 114, 115, 116,
#118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215,
#220, 223, 230]
subjects = [101]
for subject in subjects:
    file = 'data/MIT_BIH/'+str(subject)
    record = wfdb.rdsamp(file)
    annotation = wfdb.rdann(file, 'atr')

    ecg_sig = record.p_signals
    anno_index = annotation.annsamp
    anno_type = annotation.anntype
    ecg_sig=ecg_sig.astype(np.float32)
    anno_index=anno_index.astype(np.float32)
#    #extract data based on Beats
#    #find N beats
#    for i in range(len(anno_type)):
#        if (anno_type[i]=='N' and anno_index[i] > 150 and 
#             (anno_index[i]+150) < len(ecg_sig)):
#            begin_index = (anno_index[i]-150).astype(int)
#            end_index = (anno_index[i]+150).astype(int)
#            ecg_N.append(ecg_sig[begin_index:end_index, 0])
    #extract data continue
    ecg_N = ecg_sig[0:50000,0]

    
## normalize the dataset
ecg_N = np.asarray(ecg_N)
scaler1 = MinMaxScaler()
ecg_N1 = scaler1.fit_transform(ecg_N)
#ecg_N1 = scaler1.fit_transform(ecg_N.transpose()).transpose() #fit along column       
scaler2 = StandardScaler()
ecg_N2 = scaler2.fit_transform(ecg_N)
#ecg_N2 = scaler2.fit_transform(ecg_N.transpose()).transpose()
#ecg_N1 = ecg_N1.flatten()[0:50000]

# baseline remove
#baseline = np.poly1d(np.polyfit(range(len(ecg_N1)), ecg_N1, 10))
#baseline_v = baseline(range(len(ecg_N1)))

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = sparse.linalg.spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

#divide into segment for low memory contraints
baseline1 = baseline_als(ecg_N1[0:10000], 10e6, 0.05)
baseline2 = baseline_als(ecg_N1[10000:20000], 10e6, 0.05)
baseline3 = baseline_als(ecg_N1[20000:30000], 10e6, 0.05)
baseline4 = baseline_als(ecg_N1[30000:40000], 10e6, 0.05)
baseline5 = baseline_als(ecg_N1[40000:50000], 10e6, 0.05)
baseline = np.concatenate((baseline1,baseline2,baseline3,baseline4,baseline5))
ecg_N1 = ecg_N1 - baseline

n_lookback = 1
n_predict = 20

#data segmentation
x_train = []
for i in range(len(ecg_N1)-n_lookback-n_predict):
    x_train.append(ecg_N1[i:n_lookback+i])
x_train = np.asarray(x_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
y_train = np.zeros((len(x_train),n_predict))
for i in range(len(x_train)):
    y_train[i,:] = ecg_N1[n_lookback+i:n_lookback+i+n_predict]

#x_train = ecg_N1[0:-1]
#x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
#y_train = ecg_N1[1:]

# Create model
n_steps = n_lookback
n_inputs = 1
n_class = n_predict

model = Sequential()
#model.add(Dense(32, activation = 'relu', input_shape=(n_steps, n_inputs))) #n_batch*n_step*n_input batch size not in input_shape
model.add(LSTM(20, return_sequences=True, input_shape=(n_steps, n_inputs))) #output [batch_size, timesteps, units], return full output sequence (128 time step)
model.add(LSTM(20)) #output [batch_size, units], only return the last output in the output sequence corresponding to last time step
#model.add(Dense(200, activation='relu')) #output layer, output prob for every class
model.add(Dense(n_class)) #output layer, output prob for every class
model.summary()

#change learning rate
#opt = optimizers.RMSprop(lr = 0.1) #too big
#opt = optimizers.RMSprop(lr = 0.01) 

#model.compile(optimizer='rmsprop',
model.compile(optimizer = 'adam',
      loss='mean_squared_error',
      metrics=['mae'])

#batch_print_callback = callbacks.LambdaCallback(on_batch_begin=lambda batch,logs: print(batch))

model.fit(x_train, y_train,
          epochs=10,
          batch_size=1,verbose=2)
#          callbacks = [batch_print_callback])
date = str(datetime.now()).replace(" ", "").replace("-","_").replace(":","_").replace(".","_")
des = 'models/anomaly_ecg_%d_%d_%s.h5' % (n_lookback, n_predict, date)
model.save(des)
#model2 = load_model('models/anomaly_ecg_300_300_2017_08_0711_32_21_655261.h5')
#model2.layers
#model2.get_config()
#model2.summary()
    
## Result analysis
# make predictions
trainPredict = model.predict(x_train)
#testPredict = model.predict(testX)
# invert predictions
#trainPredict = scaler2.inverse_transform(trainPredict)
#trainY = scaler2.inverse_transform([y_train])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(y_train, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
#testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
recover = []       
recover_predict = []     
for i in range(int(len(y_train)/n_predict)):
    recover.extend(y_train[i*n_predict,:])
    recover_predict.extend(trainPredict[i*n_predict,:])
fig1 = plt.figure()
plt.plot(recover)
plt.plot(recover_predict)
title = 'anomaly_ecg_%d_%d_%s' % (n_lookback, n_predict, date)
#pickle.dump(fig1, open('figure/'+title+'.p', 'wb'))
## load figure
#fig = pickle.load( open( "myplot.pickle", "rb" ) )
#fig.show()
#recover_result = []
#true_result = []
#for item in trainPredict:
#    recover_result.extend(item)
#for item in y_train:
#    true_result.extend(item)
#plt.figure()
#plt.plot(recover_result)
#plt.plot(true_result)
#
#plt.figure()
#plt.plot(ecg_N)
### shift test predictions for plotting
##testPredictPlot = np.empty_like(ecg_N)
##testPredictPlot[:, :] = np.nan
##testPredictPlot[len(trainPredict)+(n_lookback*2)+1:len(ecg_N)-1, :] = testPredict
## plot baseline and predictions
#plt.figure()
#plt.plot((ecg_N))
#plt.show()