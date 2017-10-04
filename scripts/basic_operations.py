# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:47:04 2017

@author: lifeng.miao
"""

import wfdb
import numpy as np
import scipy
import matplotlib.pyplot as plt
from IPython.display import display
#create record object
ecgrecord = wfdb.rdsamp('sampledata/test01_00s', sampfrom=800, channels = [1,3])

#plot record with annotation
record = wfdb.rdsamp('sampledata/100', sampto = 3000)
annotation = wfdb.rdann('sampledata/100', 'atr', sampto = 3000)
wfdb.plotrec(record, annotation = annotation, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds', figsize = (10,4), ecggrids = 'all')
wfdb.plotann(annotation, title = None, timeunits = 'samples', returnfig = False)
plt.plot(record.p_signals[:,0])
plt.plot(annotation.annsamp)
display(record.__dict__)
wfdb.showanncodes()
#list all database and download certain database
dblist = wfdb.getdblist()
wfdb.dldatabase('ahadb', u'D:\\Python\\ECG_Machine_Learning\\wfdb-python\\data')
wfdb.dldatabase('mitdb', u'D:\\Python\\ECG_Machine_Learning\\wfdb-python\\data\\MIT_BIH')
wfdb.dldatabasefiles('cebsdb', u'D:\\Python\\ECG_Machine_Learning\\wfdb-python\\data', 
                     ['b006.hea', 'b006.dat', 'b006.atr'])

#read signal
sig, fields = wfdb.srdsamp('sampledata/test01_00s', sampfrom=800, channels = [1,3])
sig, fields = wfdb.srdsamp('data/b006', sampfrom=0, channels = [0,1,2,3])
sig, fields = wfdb.srdsamp('data/MIT_BIH/100')

ok=sig.astype(np.float32)
scipy.io.savemat('matlab/mit/100.mat',{'ecg1':ok[:,0],'ecg2':ok[:,1]})
