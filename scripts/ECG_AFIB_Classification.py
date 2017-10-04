# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:21:15 2017

@author: lifeng.miao
"""
import wfdb
import numpy as np
import matplotlib.pyplot as plt
file = "data/MIT_BIH_AFIB/04126"
record = wfdb.rdsamp(file)#, sampfrom = 100000, sampto = 120000)
annotation = wfdb.rdann(file, 'atr')#, sampfrom = 100000, sampto = 120000)
beats = wfdb.rdann(file, 'qrs')
ecg_sig = record.p_signals
anno_index = annotation.annsamp
anno_type = annotation.anntype
ecg_sig=ecg_sig.astype(np.float32)
beats_index = beats.annsamp
ibi = np.diff(beats_index)

window_size = 30
num_seg = np.floor(len(ibi)/window_size).astype(int)
#ibi_seg = np.zeros(num_seg, window_size)
afib_seg_index = []
ibi_seg_std = np.zeros(num_seg)
ibi_seg_mean = np.zeros(num_seg)
for i in range(num_seg):
    ibi_seg = ibi[i*window_size:(i+1)*window_size]
    ibi_seg_std[i] = np.std(ibi_seg)
    ibi_seg_mean[i] = np.mean(ibi_seg)
    if (ibi_seg_std[i] > 40):
        afib_seg_index.append(beats_index[i*window_size+1])
        
plt.figure()
plt.plot(ecg_sig[:,1])
plt.plot(beats_index, ecg_sig[beats_index, 1], 'r+')
plt.plot(afib_seg_index, ecg_sig[afib_seg_index, 1], 'go', linewidth=7.0)
plt.plot(anno_index, ecg_sig[anno_index, 1], 'y*', linewidth=5.0)
