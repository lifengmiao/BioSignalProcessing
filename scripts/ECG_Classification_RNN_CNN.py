# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:55:47 2017

@author: lifeng.miao
"""

import wfdb
import numpy as np
import scipy
import matplotlib.pyplot as plt
from IPython.display import display
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, TimeDistributed, Conv1D, Flatten, MaxPooling1D
from keras.utils import plot_model,to_categorical
from keras import optimizers
from keras import callbacks
from glob import glob
from sklearn.model_selection import train_test_split

x_train = []
y_train = []
ecg_V = [];ecg_S = [];ecg_F = [];ecg_N = []
cnt_N = 0
#subjects = list(range(100,110))+list(range(111,120))+list(range(121,125))
subjects = [101, 106, 108, 109, 112, 114, 115, 116,
118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215,
220, 223, 230]
for subject in subjects:
    file = 'data/MIT_BIH/'+str(subject)
    record = wfdb.rdsamp(file)
    annotation = wfdb.rdann(file, 'atr')
    
    #record = wfdb.rdsamp('data/MIT_BIH/100')
    #annotation = wfdb.rdann('data/MIT_BIH/100', 'atr')
    #wfdb.plotrec(record, annotation = annotation, title=
    #'Record 100 from MIT-BIH Arrhythmia Database', timeunits = 
    #'seconds', figsize = (10,4), ecggrids = 'all')
    
    #display(record.__dict__)
    #display(annotation.__dict__)
    
    #ecg_sig, fileds = wfdb.srdsamp('data/MIT_BIH/100')
    ecg_sig = record.p_signals
    anno_index = annotation.annsamp
    anno_type = annotation.anntype
    ecg_sig=ecg_sig.astype(np.float32)
    anno_index=anno_index.astype(np.float32)

    #find V beats
    for i in range(len(anno_type)):
        if (anno_type[i]=='V' or anno_type[i]=='E'):
#            print('subject=', subject, '; number of beats = ', i)
            begin_index = max(anno_index[i]-150, np.asarray(0)).astype(int)
            end_index = min(anno_index[i]+150, np.asarray(len(ecg_sig))).astype(int)
            ecg_V.append(ecg_sig[begin_index:end_index, 0])
        if (anno_type[i]=='A' or anno_type[i]=='a' or anno_type[i]=='J' or anno_type[i]=='S'):
#            print('subject=', subject, '; number of beats = ', i)
            begin_index = max(anno_index[i]-150, np.asarray(0)).astype(int)
            end_index = min(anno_index[i]+150, np.asarray(len(ecg_sig))).astype(int)
            ecg_S.append(ecg_sig[begin_index:end_index, 0])
        if (anno_type[i]=='F'):
#            print('subject=', subject, '; number of beats = ', i)
            begin_index = max(anno_index[i]-150, np.asarray(0)).astype(int)
            end_index = min(anno_index[i]+150, np.asarray(len(ecg_sig))).astype(int)
            ecg_F.append(ecg_sig[begin_index:end_index, 0])
        if ((anno_type[i]=='N' or anno_type[i]=='L' or anno_type[i]=='R' or 
             anno_type[i]=='e' or anno_type[i]=='j') and cnt_N < 5000):
            cnt_N = cnt_N+1
            begin_index = max(anno_index[i]-150, np.asarray(0)).astype(int)
            end_index = min(anno_index[i]+150, np.asarray(len(ecg_sig))).astype(int)
            ecg_N.append(ecg_sig[begin_index:end_index, 0])

len_max = len(sorted(ecg_F,key=len, reverse=True)[0]) #find max length of segments
ecg_F_array = [np.append(xi, [0]*(len_max-len(xi))) for xi in ecg_F] #padding 0 behind
len_max = len(sorted(ecg_N,key=len, reverse=True)[0]) #find max length of segments
ecg_N_array = [np.append(xi, [0]*(len_max-len(xi))) for xi in ecg_N] #padding 0 behind

ecg_V_array = np.asarray(ecg_V)
ecg_S_array = np.asarray(ecg_S)
ecg_F_array = np.asarray(ecg_F_array)
ecg_N_array = np.asarray(ecg_N_array)

x = np.concatenate((ecg_V_array,ecg_F_array,ecg_N_array)) #,ecg_S_array))
x = x.reshape((len(x), 300, 1))
y = np.zeros((len(x),1))
y[0:len(ecg_V_array),0] = 0
y[len(ecg_V_array):len(ecg_V_array)+len(ecg_F_array),0] = 1
y[len(ecg_V_array)+len(ecg_F_array):len(ecg_V_array)+len(ecg_F_array)+len(ecg_N_array),0] = 2
#y_train[len(ecg_V_array)+len(ecg_F_array)+len(ecg_N_array):,0] = 3

y = to_categorical(y, num_classes=3)
       
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
       
    ## Save to mat file
    #scipy.io.savemat('../matlab/mit/ecg_100.mat',{'ecg1':ecg_sig[:,0],'ecg2':ecg_sig[:,1]})
    #scipy.io.savemat('../matlab/mit/anno_index_100.mat', {'anno_index':anno_index})
    #scipy.io.savemat('../matlab/mit/anno_type_100.mat', {'anno_type':anno_type})
    
#    ecg_seg = np.zeros((len(anno_index), 300)).astype(np.float32)
#    for i in range(len(anno_index)):
#        if(anno_index[i]>150 and (anno_index[i]<len(ecg_sig)-149)):
#            begin_index = (anno_index[i]-150).astype(int)
#            end_index = (anno_index[i]+150).astype(int)
#            ecg_seg[i,:] = ecg_sig[begin_index:end_index,1]
#    
#    ecg_final = ecg_seg[2:-1,:]
#    ecg_final_type = np.asarray(anno_type[2:-1])
#    for i in range(len(ecg_final_type)):
#        if (ecg_final_type[i]=='N'):
#            ecg_final_type[i] = 1
#        else:
#            ecg_final_type[i] = 0
#    
#    ecg_final_type = ecg_final_type.astype(int)
#                    
#    x_train_new = ecg_final.reshape(ecg_final.shape[0],ecg_final.shape[1],1)
#    y_train_new = to_categorical(ecg_final_type, num_classes=2)
#    if subject == 0:
#        x_train = x_train_new
#        y_train = y_train_new
#    else:
#        x_train = np.concatenate((x_train, x_train_new))
#        y_train = np.concatenate((y_train, y_train_new))

# Create model
n_steps = 300
n_inputs = 1
n_class = 3

## RNN model
#model = Sequential()
#model.add(Dense(32, activation = 'relu', input_shape=(n_steps, n_inputs))) #n_batch*n_step*n_input batch size not in input_shape
#model.add(LSTM(32, return_sequences=True)) #output [batch_size, timesteps, units], return full output sequence (128 time step)
#model.add(LSTM(32)) #output [batch_size, units], only return the last output in the output sequence corresponding to last time step
#model.add(Dense(n_class, activation='softmax')) #output layer, output prob for every class
#model.summary()

# 1D CNN model
model = Sequential()
# apply a 1D convolution of length 3 to a sequence with 300 timesteps scalar input, output 64 output filters
model.add(Conv1D(64, 3, border_mode='same', activation = 'relu', input_shape=(n_steps, n_inputs))) # now output [batch_size, n_steps, 64]
model.add(MaxPooling1D())
model.add(Conv1D(32, 3, border_mode='same', activation = 'relu'))#border_mode='same')) # now output [batch_size, n_steps, 32]
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(n_class, activation='softmax')) #output layer, output prob for every class
model.summary()



#change learning rate
#opt = optimizers.RMSprop(lr = 0.1) #too big
opt = optimizers.RMSprop(lr = 0.01) 
#model.compile(optimizer='rmsprop',
model.compile(optimizer = opt,
      loss='categorical_crossentropy',
      metrics=['accuracy'])

batch_print_callback = callbacks.LambdaCallback(on_batch_begin=lambda batch,logs: print(batch))

model.fit(x_train, y_train,
          epochs=10,
          batch_size=1500,
          callbacks = [batch_print_callback])

model.save('models/ecg_rnn_3class.h5')

model.evaluate(x_test, y_test, batch_size = 1000)
y_predict_prob = model.predict(x_test, batch_size = 1000)
y_predict_class = np.argmax(y_predict_prob, axis = 1)