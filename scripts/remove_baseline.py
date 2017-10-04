# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:51:12 2017

@author: alireza.a
"""
from pywt import wavedec
from pywt import waverec
from pywt import wavelist

def remove_baseline(x,wavelet= wavelist(family='db')[0],level=8):
    coeffs = wavedec(x,wavelet,level=level)
    coeffs[0].fill(0)
    return waverec(coeffs, wavelet)