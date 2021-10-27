#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:58:31 2021

@author: ChristopherThomasOsborne
"""

#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_8                                                         #
# 10-19-21                                                      #
# Any other necessary information needed to navigate the file   #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-2
                          # Define step size
t = np.arange(0,20+steps,steps)       # Graph form 0 to 20
T = 8                                 # Period = 8 

# -------- Define A_k: ------------
a = np.zeros((1501,1))
for k in np.arange(1,1501):
    a[k] = 0

# -------- Define B_k: ------------
b = np.zeros((1501,1))
for k in np.arange(1,1501):
    b[k] = (-2*(np.cos(k*np.pi)-1))/(k*np.pi)
    
print ("a0 = ", a[0], " a1 =" , a[1])
print ("b1 = ", b[1], " b2 = ", b[2], " b3 = ", b[3])

# ----------- N = 1 ----------------
N = 1
val = 0
for k in np.arange(1, N+1):
    val = val + (b[k] * np.sin((k*2*np.pi*t) / T))
    
plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,val)
plt.grid()
plt.title('Fourier Transforms')
plt.ylabel('N = 1')

# ----------- N = 3 ----------------
N = 3
val = 0
for k in np.arange(1, N+1):
    val = val + (b[k] * np.sin((k*2*np.pi*t) / T))
    
plt.subplot(3,1,2)
plt.plot(t,val)
plt.grid()
plt.ylabel('N = 3')

# ----------- N = 15 ----------------
N = 15
val = 0
for k in np.arange(1, N+1):
    val = val + (b[k] * np.sin((k*2*np.pi*t) / T))
    
plt.subplot(3,1,3)
plt.plot(t,val)
plt.grid()
plt.ylabel('N = 15')

# ----------- N = 50 ----------------
N = 50
val = 0
for k in np.arange(1, N+1):
    val = val + (b[k] * np.sin((k*2*np.pi*t) / T))
    
plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,val)
plt.grid()
plt.title('Fourier Transforms')
plt.ylabel('N = 50')

# ----------- N = 150 ----------------
N = 150
val = 0
for k in np.arange(1, N+1):
    val = val + (b[k] * np.sin((k*2*np.pi*t) / T))
    
plt.subplot(3,1,2)
plt.plot(t,val)
plt.grid()
plt.ylabel('N = 150')

# ----------- N = 1500 ----------------
N = 1500
val = 0
for k in np.arange(1, N+1):
    val = val + (b[k] * np.sin((k*2*np.pi*t) / T))
    
plt.subplot(3,1,3)
plt.plot(t,val)
plt.grid()
plt.xlabel('t')
plt.ylabel('N = 1500')


