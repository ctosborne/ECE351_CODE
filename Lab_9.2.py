#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:39:36 2021

@author: ChristopherThomasOsborne
"""

#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_8                                                         #
# 10-26-21                                                      #
# Any other necessary information needed to navigate the file   #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fftpack import fft, fftshift
import matplotlib.gridspec

def fft1(f, fs):
    N = len(x1)                            
    X_fft = fft(x1)                     
    X_fft_shifted = fftshift(X_fft)   
                                                
    freq = np.arange(-N/2 , N/2) * fs/N  
                                                       
    X_mag = np.abs(X_fft_shifted)/N                   
    X_phi = np.angle(X_fft_shifted)    
            
    return freq, X_mag, X_phi

def fft_plot(t, freq, x1, X_mag, X_phi):
    
    plt.figure(figsize = (10,7))
    plt.subplot(3,1,1)
    plt.plot(t,x1)
    plt.grid()
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    plt.title('FFT')
    
    plt.subplot(3,2,3)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.ylabel('|X(f)|')

    plt.subplot(3,2,4)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlim([-2,2])
    
    plt.subplot(3,2,5)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.ylabel('/_ X(f)')
    plt.xlabel('f[Hz]')
    
    plt.subplot(3,2,6)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlim([-2,2])
    plt.xlabel('f[Hz]')
    
    plt.tight_layout()
    plt.show()
    
    return 0
fs = 100
steps = 1/fs
t = np.arange(0,2,steps)

x1 = np.cos(2*np.pi*t)
f, mag, phi = fft1(x1,fs)
fft_plot(t, f, x1, mag, phi)

x2 = 5*np.sin(2*np.pi*t)
f, mag, phi = fft1(x2, fs)
fft_plot(t, f, x2, mag, phi)

x3 = (2*np.cos(2*np.pi*2*t)-2) + (np.sin((2*np.pi*6*t)+3))**2
f, mag, phi = fft1(x3, fs)
fft_plot(t, f, x3, mag, phi)

def fft2(f, fs):
    N = len(x4)                            
    X_fft = fft(x4)                     
    X_fft_shifted = fftshift(X_fft)   
                                                
    freq = np.arange(-N/2 , N/2) * fs/N  
                                                       
    X_mag = np.abs(X_fft_shifted)/N                   
    X_phi = np.angle(X_fft_shifted)    
        
    for i in range(len(X_mag)):
        if X_mag[i] < 1e-10: 
            X_phi[i] = 0
    
    return freq, X_mag, X_phi

def fft_plot2(t, freq, x1, X_mag, X_phi):
    
    plt.figure(figsize = (10,7))
    plt.subplot(3,1,1)
    plt.plot(t,x1)
    plt.grid()
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    plt.title('FFT')
    
    plt.subplot(3,2,3)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.ylabel('|X(f)|')

    plt.subplot(3,2,4)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlim([-15,15])
    
    plt.subplot(3,2,5)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.ylabel('/_ X(f)')
    plt.xlabel('f[Hz]')
    
    plt.subplot(3,2,6)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlim([-2,2])
    plt.xlabel('f[Hz]')
    
    plt.tight_layout()
    plt.show()
    
    return 0

x4 = np.cos(2*np.pi*t)
f, mag, phi = fft2(x4,fs)
fft_plot2(t, f, x4, mag, phi)

x5 = 5*np.sin(2*np.pi*t)
f, mag, phi = fft2(x5, fs)
fft_plot2(t, f, x5, mag, phi)

x6 = (2*np.cos(2*np.pi*2*t)-2) + (np.sin((2*np.pi*6*t)+3))**2
f, mag, phi = fft2(x6, fs)
fft_plot2(t, f, x6, mag, phi)

t = np.arange(0, 16, steps)
T = 8
x7 = 0

for k in np.arange(1, 15+1):
    b = 2/(k*np.pi)*(1-np.cos(k*np.pi))
    x = b*np.sin(k*(2*np.pi/T)*t)
    x7 += x
    
f, mag, phi = fft2(x7, fs)
fft_plot2(t, f, x7, mag, phi)





