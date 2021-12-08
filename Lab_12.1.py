#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:51:36 2021

@author: ChristopherThomasOsborne
"""
#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_12                                                        #
# 12-6-21                                                       #
#                                                               #
#                                                               #
#################################################################
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
import numpy as np
import scipy.signal as sig

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

# unfiltered signals - (sensor_sig):
plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal ')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()


# -- Fast Fourier Transform:
def FFT_X(X, fs):
    N = len(X)                            
    X_fft = fft(X)                     
    X_fft_shifted = fftshift(X_fft)   
                                                
    freq = np.arange(-N/2 , N/2) * fs/N  
                                                       
    X_mag = np.abs(X_fft_shifted)/N                   
    X_phi = np.angle(X_fft_shifted)    
        
    for i in range(len(X_mag)):
        if X_mag[i] < 1e-10: 
            X_phi[i] = 0
    
    return freq, X_mag, X_phi

fs = 1e6
x1 =  sensor_sig
f, mag, phi = FFT_X(x1,fs)

# Unfiltered Signal - FFT(sensor_sig):
plt.figure(figsize = (10, 7))
plt.plot(f, mag)
plt.semilogx(f, mag)
plt.grid()
plt.title('Unfiltered Signal - FFT(sensor_sig): 0 to 1e6 Hz')
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([1,1e6])
plt.ylabel('|mag|')
plt.xlabel('f[Hz]')
plt.show()

plt.figure(figsize = (10, 7))
plt.plot(f, mag)
plt.semilogx(f, mag)
plt.grid()
plt.title('Unfiltered Signal - FFT(sensor_sig)- (1 to 1e3 Hz):')
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([1e1,1e3])
plt.ylabel('|mag|')
plt.xlabel('f[Hz]')
plt.show()

plt.figure(figsize = (10, 7))
plt.plot(f, mag)
plt.semilogx(f, mag)
plt.grid()
plt.title('Unfiltered Signal - FFT(sensor_sig)- (1800 to 2000 Hz):')
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([1700,2100])
plt.ylabel('|mag|')
plt.xlabel('f[Hz]')
plt.show()

plt.figure(figsize = (10, 7))
plt.plot(f, mag)
plt.semilogx(f, mag)
plt.grid()
plt.title('Unfiltered Signal - FFT(sensor_sig)- (1e4 to 1e6 Hz):')
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([1e4,1e6])
plt.ylabel('|mag|')
plt.xlabel('f[Hz]')
plt.show()


# Circuit: Series RLC Circuit
steps = 1e3
R = 100
Band_Width = 1256.6366
W_o=11938.053
L = (R/Band_Width)
C = (1/((W_o**2)*L))

print('R =', R)
print('L =', L)
print('C =', C)

num = [0,R/L,0]
den = [1,R/L,1/(L*C)]

# Z-Trnsform Of The RLC Circuit:
# Filtered Signal - FFT(sensor_sig):
z_num, z_den = sig.bilinear(num, den, fs)
X_Filter = sig.lfilter(z_num, z_den, x1)

_, X_Filter_MAG, X_Filter_PHI = FFT_X(X_Filter, fs)


plt.figure(figsize = (10,7))
plt.semilogx(f, X_Filter_MAG)
plt.ylim([0,1])
plt.xlim([1,1e6])
plt.grid()
plt.title('Filtered Signal - FFT(sensor_sig)- (0 to 1e6 Hz:)')
plt.xlabel('f[Hz]')
plt.ylabel('Magnitude')
plt.show()

plt.figure(figsize = (10,7))
plt.semilogx(f, X_Filter_MAG)
plt.ylim([0,.01])
plt.xlim([1,1000])
plt.grid()
plt.title('Filtered Signal - FFT(sensor_sig)- (1 to 1e3 Hz):')
plt.xlabel('f[Hz]')
plt.ylabel('Magnitude')
plt.show()

plt.figure(figsize = (10,7))
plt.semilogx(f, X_Filter_MAG)
plt.ylim([0,1])
plt.xlim([1e3,1e4])
plt.grid()
plt.title('Filtered Signal - FFT(sensor_sig)- (1800 to 2000 Hz):')
plt.xlabel('f[Hz]')
plt.ylabel('Magnitude')
plt.show()

plt.figure(figsize = (10,7))
plt.semilogx(f, X_Filter_MAG)
plt.ylim([0,.01])
plt.xlim([1e4,1e6])
plt.grid()
plt.title('Filtered Signal - FFT(sensor_sig)- (1e4 to 1e6 Hz):')
plt.xlabel('f[Hz]')
plt.ylabel('Magnitude')
plt.show()



#  Bode Plots:
w = np.arange(1e3, 1e6 + steps, steps)
w, H_mag, H_deg = sig.bode((num, den), w)


plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w/(2*np.pi), H_mag)
plt.grid()
plt.ylim([-50,3])
plt.xlim([1e2, 1e5])
plt.ylabel('Magnitude')
plt.title('Bode Plot - (0 to 1e6 Hz)')
plt.subplot(2,1,2)
plt.semilogx(w/(2*np.pi), H_deg*2*np.pi)
plt.grid()
plt.xlim([1e2, 1e5])
plt.ylabel('Degrees')
plt.xlabel('Frequency [Hz]')
plt.show()

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w/(2*np.pi), H_mag)
plt.grid()
plt.xlim([1, 1e3])
plt.ylabel('Magnitude')
plt.title('Bode Plot - (1 to 1e3 Hz)')
plt.subplot(2,1,2)
plt.semilogx(w/(2*np.pi), H_deg*2*np.pi)
plt.grid()
plt.xlim([1, 1e3])
plt.ylabel('Degrees')
plt.xlabel('Frequency [Hz]')
plt.show()

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w/(2*np.pi), H_mag)
plt.grid()
plt.ylim([-.4,0])
plt.xlim([1800, 2000])
plt.ylabel('Magnitude')
plt.title('Bode Plot - (1800 to 2000 Hz)')
plt.subplot(2,1,2)
plt.semilogx(w/(2*np.pi), H_deg*2*np.pi)
plt.grid()
plt.xlim([1700, 2100])
plt.ylabel('Degrees')
plt.xlabel('Frequency [Hz]')
plt.show()

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w/(2*np.pi), H_mag)
plt.grid()
plt.xlim([1e4, 1e6])
plt.ylabel('Magnitude')
plt.title('Bode Plot - (1e4 to 1e6 Hz)')
plt.subplot(2,1,2)
plt.semilogx(w/(2*np.pi), H_deg*2*np.pi)
plt.grid()
plt.xlim([1e4, 1e6])
plt.ylabel('Degrees')
plt.xlabel('Frequency [Hz]')
plt.show()


#  Filtered Signal - (sensor_sig):
plt.figure(figsize = (10,7))
plt.plot(t, X_Filter)
plt.grid()
plt.title('Filtered Signal - (sensor_sig):')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

