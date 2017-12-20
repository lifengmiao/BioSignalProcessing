# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:21:15 2017

@author: lifeng.miao
"""
import wfdb

import numpy as np
import matplotlib.pyplot as plt
import glob
    
def SampEn(U, m, r):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r])) - 1 for x_i in x]
        return sum(C)

    N = len(U)

    return np.log2(_phi(m)/_phi(m+1))

def StatisticAnalysisofIBI(ibi_out, window_size):
    #Statistic Analysis of AFIB ibi
    ibiTPR = []
    ibiRMSSD = []
    ibiShannonEntropy = []
    ibiSampleEntropy = []
    for i in np.arange(0, len(ibi_out)-window_size+1, window_size):
#        print(i)
        ibi_seg = ibi_out[i:i+window_size]
        ibiSegLen = len(ibi_seg)
        ibiMean = np.mean(ibi_seg)
#        ibiRStd = np.std(ibi_seg)
        #Turning point
        ibiTpCnt = 0
        for kk in range(1,len(ibi_seg)-1):
            if(ibi_seg[kk-1]<ibi_seg[kk]>ibi_seg[kk+1] or ibi_seg[kk-1]>ibi_seg[kk]<ibi_seg[kk+1]):
                ibiTpCnt += 1
        ibiTPR.append(ibiTpCnt/ibiSegLen)
        #Root mean square of successive difference (RMSSD)
        ibiRMSSD.append(np.sqrt(np.sum([x**2 for x in np.diff(ibi_seg)])/(ibiSegLen-1))/ibiMean)
        #Shannon Entropy
        ibi_seg_tmp = ibi_seg
        for jj in range(4):
            ibi_seg_tmp = removeMinMax(ibi_seg_tmp)
        h = np.histogram(ibi_seg_tmp,48)
        p = h[0]/(ibiSegLen-8)
        ibiShannonEntropy_tmp = 0
        for item in p:
            if(item!=0):
                ibiShannonEntropy_tmp += item*np.log2(item)/np.log2(1/48)
        ibiShannonEntropy.append(ibiShannonEntropy_tmp)
        #Sample Entropy
        ibiSampleEntropy_tmp = SampEn(ibi_seg, 2, 0.15*250)
        ibiSampleEntropy.append(ibiSampleEntropy_tmp)
    return ibiTPR,ibiRMSSD,ibiShannonEntropy,ibiSampleEntropy

def removeMinMax(seg):
    seg_tmp = np.delete(seg, seg.argmin())
    seg_tmp = np.delete(seg_tmp, seg_tmp.argmax())
    return seg_tmp

'''
-------------------------------------------------------------------------------
AF IBI extraction
-------------------------------------------------------------------------------
'''
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
    record = wfdb.rdsamp(file)#, sampfrom = 100000, sampto = 120000)
    annotation = wfdb.rdann(file, 'atr')#, sampfrom = 100000, sampto = 120000)
    beats = wfdb.rdann(file, 'qrs')
    ecg_sig = record.p_signals
    anno_index = annotation.sample
    anno_type = annotation.subtype
    anno_AF = annotation.aux_note
#    print(anno_AF)
    ecg_sig=ecg_sig.astype(np.float32)
    beats_sample = beats.sample
    ibi = np.diff(beats_sample)
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
ibi_out_AF= []
for item in ibi_AF:
    ibi_out_AF = np.append(ibi_out_AF, item)

#'''
#-------------------------------------------------------------------------------
#   Normal IBI extraction
#-------------------------------------------------------------------------------
#'''
#ibi_out_NSR = []
#for filename in glob.iglob('data/MIT_BIH_NSR/*.hea'):
#    print(filename)
#    aa = ''
#    for s in filename:
#        if s.isdigit():
#            aa = aa+s
#    file = 'data/MIT_BIH_NSR/'+aa
#    record = wfdb.rdsamp(file)
#    annotation = wfdb.rdann(file, 'atr')
#    beats_raw = annotation.sample
#    anno_type = annotation.subtype
#    anno_note = annotation.aux_note
#    anno_symbol = annotation.symbol
#    ecg_sig = record.p_signals
#    NSR_indx = np.squeeze(np.array(np.where(np.array(anno_symbol)=='N'))) #find NSR beats
#    bb = np.squeeze(np.diff(NSR_indx)) 
#    gap_indx = np.squeeze(np.array(np.where(bb>1))) #find gap indx in NSR beats
#    beat_range = [] # (start_indx, end_indx) of NSR beats segments
#    for i in range(len(gap_indx)):
#        if i==0:
#            beat_range.append((NSR_indx[0], NSR_indx[gap_indx[i]]))
#        else:
#            beat_range.append((NSR_indx[gap_indx[i-1]+1], NSR_indx[gap_indx[i]]))
#
#    for start,end in beat_range:
#        beats_good = beats_raw[start:end+1]
#        ibi_good = np.diff(beats_good)
#        ibi_out_NSR = np.append(ibi_out_NSR, ibi_good)


'''------------------------------------
Featrue extration
'''
ibi_out_NSR = np.load('data/MIT_BIH_NSR/NSR_Beats.npy')
windowSize = 60
ibiTPR_AF, ibiRMSSD_AF, ibiShannonEntropy_AF, ibiSampleEntropy_AF = StatisticAnalysisofIBI(ibi_out_AF, windowSize)
ibiTPR_NSR, ibiRMSSD_NSR, ibiShannonEntropy_NSR, ibiSampleEntropy_NSR = StatisticAnalysisofIBI(ibi_out_NSR, windowSize)
np.save('data/MIT_BIH_NSR/ibiTPR_NSR.npy',ibiTPR_NSR)
np.save('data/MIT_BIH_NSR/ibiRMSSD_NSR.npy',ibiRMSSD_NSR)
np.save('data/MIT_BIH_NSR/ibiShannonEntropy_NSR.npy',ibiShannonEntropy_NSR)
np.save('data/MIT_BIH_NSR/ibiSampleEntropy_NSR.npy',ibiSampleEntropy_NSR)
np.save('data/MIT_BIH_NSR/ibiTPR_AF.npy',ibiTPR_AF)
np.save('data/MIT_BIH_NSR/ibiRMSSD_AF.npy',ibiRMSSD_AF)
np.save('data/MIT_BIH_NSR/ibiShannonEntropy_AF.npy',ibiShannonEntropy_AF)
np.save('data/MIT_BIH_NSR/ibiSampleEntropy_AF.npy',ibiSampleEntropy_AF)
###############################################################################
## Histogram plot
###############################################################################

n, bins, patches = plt.hist(np.array(ibiRMSSD_NSR), 500, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = plt.hist(np.array(ibiRMSSD_AF), 500, normed=1, facecolor='red', alpha=0.75)
#l = plt.plot(bins[1::], n, 'r--', linewidth=1)
plt.xlabel('ibiRMSSD')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ RMSSD:}\ \bar=100,\ \alpha=0.75$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

plt.figure()
n, bins, patches = plt.hist(np.array(ibiShannonEntropy_NSR), 500, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = plt.hist(np.array(ibiShannonEntropy_AF), 500, normed=1, facecolor='red', alpha=0.75)
#l = plt.plot(bins[1::], n, 'r--', linewidth=1)
plt.xlabel('ibiShannonEntropy')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ ShannonEntropy:}\ bar=100,\ \alpha=0.75$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

plt.figure()
n, bins, patches = plt.hist(np.array(ibiSampleEntropy_NSR), 500, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = plt.hist(np.array(ibiSampleEntropy_AF), 500, normed=1, facecolor='red', alpha=0.75)
#l = plt.plot(bins[1::], n, 'r--', linewidth=1)
plt.xlabel('ibiSampleEntropy')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ SampleEntropy:}\ bar=100,\ \alpha=0.75$')
plt.axis([0, 1.75, 0, 10])
plt.grid(True)
plt.show()


##IBI vs onset of AF 
#plt.figure()
#plt.plot(beats_sample[1::], ibi)
#plt.plot(anno_index, ecg_sig[anno_index, 1], 'y*', linewidth=5.0)

#plt.plot(ecg_sig[:,1])
#plt.plot(beats_sample, ecg_sig[beats_sample, 1], 'r+')
#plt.plot(afib_seg_index, ecg_sig[afib_seg_index, 1], 'go', linewidth=7.0)

