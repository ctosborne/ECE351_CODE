#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:22:13 2021

@author: ChristopherThomasOsborne
"""

#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_2                                                         #
# 9-7-21                                                        #
# Any other necessary information needed to navigate the file   #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
steps = 1e-2 # Define step size
t1 = np.arange(-5 , 10 + steps , steps ) # Add a step size to make sure the
                                         # plot includes 5.0. Since np. arange () only
                                         # goes up to , but doesn ’t include the
                                         # value of the second argument
print('Number of elements : len(t) = ', len( t1 ) , '\nFirst Element : t[0] = ', t1 [0] ,
      '\nLast Element : t[len(t) - 1] = ', t1 [len( t1 ) - 1])


# Notice the array might be a different size than expected since Python starts
# at 0. Then we will use our knowledge of indexing to have Python print the
# first and last index of the array . Notice the array goes from 0 to len () - 1    

# --- User - Defined Function ---
def step(t1):
    y = np.zeros( t1.shape )
    for i in range(len( t1 ) ):
        if t1[i] >= 0:     
            y[i] = 1
        else:
            y[i] = 0
    return y

def ramp(t1):
    z = np.zeros( t1.shape )
    for i in range(len( t1 )):
        if t1[i] <= 0:
            z[i] = 0
        else:
            z[i] = t1[i]
    return z

def step_ramp(t1):
    yz = ramp(t1)-ramp(t1-3)+5*step(t1-3)-2*step(t1-6)-2*ramp(t1-6)
    return yz


yz = step_ramp( t1 ) 
plt.figure( figsize = (10, 10) )
plt.subplot(2 , 1 , 1)
plt.plot(-t1-4, yz )
plt.grid()
plt.ylabel('u(t) and r(t)  ')
plt.title('time-shift operations f(-t − 4)  ')
  

