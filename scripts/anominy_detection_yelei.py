# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:42:08 2017

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
sig, fields = wfdb.srdsamp('data/m008', sampfrom=0, channels = [0])
sig1=scipy.signal.resample(sig,int(sig.size/50))
sig2=sig1[0:5000]
dataset = sig2.astype('float32')
np.random.seed(7)

#dataframe = pd.read_csv('C:\\Users\yelei.li\Downloads\international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataset = dataframe.values
#dataset = dataset.astype('float32')



## normalize the dataset
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

#x_train,x_test = train_test_split(dataset,test_size=.3,random_state=32)
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
x_train, x_test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back,pred_win):
               dataX, dataY = [], []
               for i in range(len(dataset)-look_back-pred_win-1):
                              a = dataset[i:(i+look_back), 0]
                              dataX.append(a)
                              dataY.append(dataset[i+look_back:(i+pred_win)+look_back, 0])
               return np.array(dataX), np.array(dataY)

look_back = 200
pred_win  = 20

trainX, trainY = create_dataset(x_train, look_back,pred_win)
testX, testY = create_dataset(x_test, look_back,pred_win)



# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(look_back, input_shape=(1, look_back)))
model.add(Dense(pred_win))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot((dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
