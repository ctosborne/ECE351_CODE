#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:44:13 2021

@author: ChristopherThomasOsborne
"""

#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_10                                                         #
# 11-1-21                                                      #
# Any other necessary information needed to navigate the file   #
#                                                               #
#################################################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

def phase(H_deg):
    for i in range(len(H_deg)):
        if H_deg[i] > 90:
            H_deg[i] = H_deg[i] - 180
    return H_deg

steps = 1e3
R = 1e3
L = 27e-3
C = 100e-9

w = np.arange(1e3, 1e6 + steps, steps)
H_mag = (20*np.log10((w/(R*C))/(np.sqrt(w**4  + (1/(R*C)**2 - 2/(L*C))*w**2 + (1/(L*C))**2))))
H_deg = (np.pi/2 - np.arctan((w/(R*C))/(-w**2 + 1/(L*C)))) * 180/np.pi
H_deg = phase(H_deg)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w, H_mag)
plt.grid()
plt.ylabel('Magnitude in dB')
plt.title('Part 1 - Task 1')
plt.subplot(2,1,2)
plt.semilogx(w, H_deg)
plt.yticks([-90, -45, 0, 45, 90])
plt.ylim([-90,90])
plt.grid()
plt.ylabel('Phase in degrees')
plt.xlabel('Frequency in rad/s')
plt.show()

num = [0,1/(L*C),0]
den = [1,1/(R*C),1/(L*C)]

w, H_mag, H_deg = sig.bode((num, den), w)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w, H_mag)
plt.grid()
plt.xlim([1e3, 1e6])
plt.title('Part 1 - Task 2')
plt.subplot(2,1,2)
plt.semilogx(w, H_deg)
plt.grid()
plt.xlim([1e3, 1e6])
plt.ylim([-90,90])
plt.xlabel('Frequency in rad/s')
plt.show()

# part 1 - Task 3:
sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, Plot = True)

# part 2 - Task 1:
fs = 50000*2*np.pi
steps = 1/fs
t = np.arange(0, 0.01 + steps, steps)
x_t = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure(figsize = (10,7))
plt.plot(t, x_t)
plt.grid()
plt.title('Part 2 - Task 1')
plt.xlabel('t')
plt.ylabel('Magnitude')
plt.show()

# part 2 - Task 2:
z_xt, p_xt = sig.bilinear(num, den, fs)

y_t = sig.lfilter(z_xt, p_xt, x_t)

plt.figure(figsize = (10,7))
plt.plot(t, y_t)
plt.grid()
plt.title('')
plt.xlabel('t')
plt.ylabel('Magnitude')
plt.show()