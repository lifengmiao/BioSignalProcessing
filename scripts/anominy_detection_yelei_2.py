# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:07:02 2017

@author: lifeng.miao
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import scipy

#dataset = pd.read_csv('C:\\Users\yelei.li\Downloads\international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#plt.plot(dataset)
#plt.show()
import wfdb
ecg, fields = wfdb.srdsamp('C:\work\documentation\BCG\physionet\m008', sampfrom=0, channels = [0])
ecg=scipy.signal.resample(ecg,int(ecg.size/50))
scg, fields = wfdb.srdsamp('C:\work\documentation\BCG\physionet\m008', sampfrom=0, channels = [3])
scg=scipy.signal.resample(scg,int(scg.size/50))

ecg=ecg[0:50000]
ecg = ecg.astype('float32')

scg=scg[0:50000]
scg = scg.astype('float32')
np.random.seed(7)

#dataframe = pd.read_csv('C:\\Users\yelei.li\Downloads\international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataset = dataframe.values
#dataset = dataset.astype('float32')



## normalize the dataset
scaler = MinMaxScaler()
ecg = scaler.fit_transform(ecg)
scg = scaler.fit_transform(scg)


#x_train,x_test = train_test_split(dataset,test_size=.3,random_state=32)
train_size = int(len(ecg) * 0.67)
test_size = len(ecg) - train_size
ecg_train, ecg_test = ecg[0:train_size,:], ecg[train_size:len(ecg),:]
scg_train, scg_test = scg[0:train_size,:], scg[train_size:len(scg),:]

# convert an array of values into a dataset matrix
def create_dataset(ecg, scg, n_window,timestep):
               dataX, dataY = [], [];N_window =  int((len(ecg)-n_window-1)/timestep)
               for i in range(N_window):
                              a = ecg[i*timestep:(i*timestep+n_window), 0];
                              b = scg[i*timestep:(i*timestep+n_window), 0];      
                              dataX.append(a)
                              dataY.append(b)
               return np.array(dataX), np.array(dataY)

look_back = 100
n_window  = 100
timestep  = 50

trainX, trainY = create_dataset(ecg_train, scg_train,n_window,timestep)
testX, testY = create_dataset(ecg_test, scg_test,n_window,timestep)



# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0],  trainX.shape[1] ,1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1] ,1))

model = Sequential()
model.add(LSTM(20, input_shape=(look_back,1)))
model.add(Dense(look_back))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=90, batch_size=1, verbose=1)
#
#
## make predictions
#trainPredict = model.predict(trainX)
#testPredict = model.predict(testX)
## invert predictions
##trainPredict = scaler.inverse_transform(trainPredict)
##trainY = scaler.inverse_transform([trainY])
##testPredict = scaler.inverse_transform(testPredict)
##testY = scaler.inverse_transform([testY])
## calculate root mean squared error
##trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
##print('Train Score: %.2f RMSE' % (trainScore))
##testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
##print('Test Score: %.2f RMSE' % (testScore))
#
#
#
## shift train predictions for plotting
#trainPredictPlot = np.empty_like(dataset)
#trainPredictPlot[:, :] = np.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
## shift test predictions for plotting
#testPredictPlot = np.empty_like(dataset)
#testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
## plot baseline and predictions
#plt.plot((dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()
