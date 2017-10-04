# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:01 2017

@author: lifeng.miao
"""
import pandas as pd
import matplotlib.pyplot as plt
from baseline_als import baseline_als

# read csv 
data = pd.read_csv('C:\Work\Data\ECG AF\ecg_normal.csv')       
data = data.values
data = data[:,0] #change to 1D
# baseline remove
baseline = baseline_als(data, 10e6, 0.05)
data_rmBaseline = data-baseline
# plot
plt.figure()
plt.plot(data)
plt.plot(baseline)
plt.plot(data_rmBaseline)
