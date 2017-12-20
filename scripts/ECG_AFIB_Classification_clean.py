# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:51:23 2017

@author: lifeng.miao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:21:15 2017

@author: lifeng.miao
"""
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import glob

def dataExtractMITAFIB(file):
    record = wfdb.rdsamp(file)
    annotation = wfdb.rdann(file, 'atr')
    beats = wfdb.rdann(file, 'qrs')
    ecg_sig = record.p_signals
    anno_index = annotation.sample
    anno_type = annotation.subtype
    anno_note = annotation.aux_note
    anno_symbol = annotation.symbol
    beats_raw = beats.sample
    return ecg_sig,anno_index,anno_type,anno_note,anno_symbol,beats_raw

def dataExtractMITNSR(file):
    record = wfdb.rdsamp(file)
    annotation = wfdb.rdann(file, 'atr')
    beats_raw = annotation.sample
    anno_type = annotation.subtype
    anno_note = annotation.aux_note
    anno_symbol = annotation.symbol
    ecg_sig = record.p_signals
    return ecg_sig,anno_index,anno_type,anno_note,anno_symbol,beats_raw

ibi_AF = []
for filename in glob.iglob('data/MIT_BIH_AFIB/*.hea'):
    print(filename)
    aa = ''
    for s in filename:
        if s.isdigit():
            aa = aa+s
    if (aa=='04936' or aa=='05091'):
        continue
    file = 'data/MIT_BIH_AFIB/'+aa
    ecg_sig,anno_index,anno_type,anno_AF,beats_sample = dataExtractMITAFIB(file)
    ibi = np.diff(beats_sample)

    #window_size = 30
    #num_seg = np.floor(len(ibi)/window_size).astype(int)
    ##ibi_seg = np.zeros(num_seg, window_size)
    #afib_seg_index = []
    #ibi_seg_std = np.zeros(num_seg)
    #ibi_seg_mean = np.zeros(num_seg)
    #for i in range(num_seg):
    #    ibi_seg = ibi[i*window_size:(i+1)*window_size]
    #    ibi_seg_std[i] = np.std(ibi_seg)
    #    ibi_seg_mean[i] = np.mean(ibi_seg)
    #    if (ibi_seg_std[i] > 40):
    #        afib_seg_index.append(beats_sample[i*window_size+1])
            
    #Extract Beats from AF episodes
    indx_start = []
    indx_end = []
    for i in range(len(anno_AF)):
        if aa=='04126':
            if anno_AF[i].find('AF')!=-1:
                indx_start.append(anno_index[i])
                if i == len(anno_AF)-1:
                    indx_end.append(len(ecg_sig)-1)
                else:
                    indx_end.append(anno_index[i+1]-1)
        else:
            if anno_AF[i].find('AFIB')!=-1:
                indx_start.append(anno_index[i])
                if i == len(anno_AF)-1:
                    indx_end.append(len(ecg_sig)-1)
                else:
                    indx_end.append(anno_index[i+1]-1)
    indx = zip(indx_start, indx_end)
    for start,end in indx:
        indx_in_range = np.where((beats_sample>start) & (beats_sample<end))
        ibi_AF.append(np.diff(beats_sample[indx_in_range]))
ibi_out = []
for item in ibi_AF:
    ibi_out = np.append(ibi_out, item)
    
def StatisticAnalysisofIBI(ibi_out):
    #Statistic Analysis of AFIB ibi
    ibiTPR = []
    ibiRMSSD = []
    ibiShannonEntropy = []
    for i in range(len(ibi_out)-60+1):
        ibi_seg = ibi_out[i:i+60]
        ibiSegLen = len(ibi_seg)
        ibiMean = np.mean(ibi_seg)
        ibiRStd = np.std(ibi_seg)
        #Turning point
        ibiTpCnt = 0
        for kk in range(1,len(ibi_seg)-1):
            if(ibi_seg[kk-1]<ibi_seg[kk]>ibi_seg[kk+1] or ibi_seg[kk-1]>ibi_seg[kk]<ibi_seg[kk+1]):
                ibiTpCnt += 1
        ibiTPR.append(ibiTpCnt/ibiSegLen)
        #Root mean square of successive difference (RMSSD)
        ibiRMSSD.append(np.sqrt(np.sum([x**2 for x in np.diff(ibi_seg)])/(ibiSegLen-1))/ibiMean)
        #Shannon Entropy
        ibi_seg_tmp = np.delete(ibi_seg, ibi_seg.argmin())
        ibi_seg_tmp = np.delete(ibi_seg_tmp, ibi_seg_tmp.argmax())
        h = np.histogram(ibi_seg_tmp,16)
        p = h[0]/(ibiSegLen-2)
        ibiShannonEntropy_tmp = 0
        for item in p:
            if(item!=0):
                ibiShannonEntropy_tmp += item*np.log(item)/np.log(1/16)
        ibiShannonEntropy.append(ibiShannonEntropy_tmp)
    return (ibiTPR,ibiRMSSD,ibiShannonEntropy)


'''----------------------------------------------------------------------------
   Normal data
----------------------------------------------------------------------------'''
ibi_out_NSR = []
for filename in glob.iglob('data/MIT_BIH_NSR/*.hea'):
    print(filename)
    aa = ''
    for s in filename:
        if s.isdigit():
            aa = aa+s
    file = 'data/MIT_BIH_NSR/'+aa
    ecg_sig,anno_index,anno_type,anno_note,anno_symbol,beats_raw = dataExtractMITNSR
    NSR_indx = np.squeeze(np.array(np.where(np.array(anno_symbol)=='N'))) #find NSR beats
    bb = np.squeeze(np.diff(NSR_indx)) 
    gap_indx = np.squeeze(np.array(np.where(bb>1))) #find gap indx in NSR beats
    beat_range = [] # (start_indx, end_indx) of NSR beats segments
    for i in range(len(gap_indx)):
        if i==0:
            beat_range.append((NSR_indx[0], NSR_indx[gap_indx[i]]))
        else:
            beat_range.append((NSR_indx[gap_indx[i-1]+1], NSR_indx[gap_indx[i]]))

    for start,end in beat_range:
        beats_good = beats_raw[start:end+1]
        ibi_good = np.diff(beats_good)
        ibi_out_NSR = np.append(ibi_out_NSR, ibi_good)

ibiTPR_NSR, ibiRMSSD_NSR, ibiShannonEntropy_NSR = StatisticAnalysisofIBI(ibi_out_NSR)

###############################################################################
## Histogram plot
###############################################################################

n, bins, patches = plt.hist(np.array(ibiRMSSD), 500, normed=1, facecolor='green', alpha=0.75)
l = plt.plot(bins[1::], n, 'r--', linewidth=1)
plt.xlabel('ibiRMSSD')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ RMSSD:}\ \bar=100,\ \alpha=0.75$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(np.array(ibiShannonEntropy), 500, normed=1, facecolor='green', alpha=0.75)
l = plt.plot(bins[1::], n, 'r--', linewidth=1)
plt.xlabel('ibiShannonEntropy')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ ShannonEntropy:}\ bar=100,\ \alpha=0.75$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()




##IBI vs onset of AF 
#plt.figure()
#plt.plot(beats_sample[1::], ibi)
#plt.plot(anno_index, ecg_sig[anno_index, 1], 'y*', linewidth=5.0)

#plt.plot(ecg_sig[:,1])
#plt.plot(beats_sample, ecg_sig[beats_sample, 1], 'r+')
#plt.plot(afib_seg_index, ecg_sig[afib_seg_index, 1], 'go', linewidth=7.0)

