#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:55:48 2021

@author: ChristopherThomasOsborne
"""

#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_5                                                         #
# 9-28-21                                                       #
# Any other necessary information needed to navigate the file   #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 1e-8 # Define step size
t1 = np.arange(0 , 0.0012 + steps , steps ) 
print('Number of elements : len(t) = ', len( t1 ) , '\nFirst Element : t[0] = ', t1 [0] ,
      '\nLast Element : t[len(t) - 1] = ', t1 [len( t1 ) - 1])

# --- User - Defined Function ---
def step(t):
    y = np.zeros( t.shape )
    for i in range(len( t ) ):
        if t[i] >= 0:     
            y[i] = 1
        else:
            y[i] = 0
    return y

def ramp(t):
    z = np.zeros( t.shape )
    for i in range(len( t )):
        if t[i] <= 0:
            z[i] = 0
        else:
            z[i] = t[i]
    return z

ht = 10355.606*np.exp(-5000*t1)*np.cos(18584.143*t1 + 0.262822971045)

HS = ([0,10000,0],[1,10000,3.703703e8])
tout , yout = sig . impulse (((HS)) , T = t1)

####################################################################
plt.figure( figsize = (10, 7) )
plt.subplot(2 , 1 , 1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('H(s)')
plt.title('Libary Convolve') 

plt.subplot(2 , 1 , 2)
plt.plot( t1, ht)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel ('convolve by Hand')
####################################################################