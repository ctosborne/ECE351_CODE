#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 19:34:24 2021

@author: ChristopherThomasOsborne
"""

#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_3                                                         #
# 9-14-21                                                       #
# Any other necessary information needed to navigate the file   #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 1e-2 # Define step size
t1 = np.arange(-10 , 10 + steps , steps ) 
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


def f1(t):
     a = np.exp(-2*t)*(step(t)-step(t-3))
     return a
 
def f2(t):
     b = step(t-2)-step(t-6)
     return b
 
def f3(t):  
     w = 0.25*2*np.pi                                
     c = np.cos(w*t)*step(t)
     return c

def H1(t):
    q = (0.5-0.5*np.exp(-2*t))*step(t)-(0.5-0.5*np.exp(-2*(t-3)))*step(t-3)
    return q

def H2(t):
    r = ramp(t-2)-ramp(t-6)-ramp(t-12)
    return r

def H3(t):
    k = 0.25*2*np.pi
    u = (36.475626*np.sin(k*t))*step(t)
    return u
 
def conv(f1,f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2-1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1-1)))
    result = np.zeros(f1Extended.shape)
    
    for i in range(Nf2 + Nf1-2):
        result[i] = 0
        for j in range(Nf1):
            if ((i - j + 1) > 0):
                try:
                    result[i] += f1Extended[j] * f2Extended[i - j + 1]
                except:
                    print(i,j)
    return (result * steps)

t2= np.arange(2 * t1[0], 2 * t1 [len(t1) - 1] + steps, steps)

####################################################################
a = f1( t1 ) 
plt.figure( figsize = (10, 7) )
plt.subplot(3 , 1 , 1)
plt.plot(t1 , a )
plt.grid()
plt.ylabel('f1(t) ')
plt.title('f1(t), f2(t), f3(t) ')
   
b = f2( t1 ) 
plt.subplot(3 , 1 , 2)
plt.plot(t1 , b )
plt.grid()
plt.ylabel('f2(t)')


c = f3( t1 ) 
plt.subplot(3 , 1 , 3)
plt.plot(t1 , c )
plt.grid()
plt.ylabel('f3(t) ')
####################################################################

####################################################################
y = step(t1)
q = H1(t1)
plt.figure( figsize = (10, 7) )
plt.subplot(2 , 1 , 1)
plt.plot(t2 , conv(f1(t1),y))
plt.grid()
plt.ylabel('f1(t) ')
plt.title('Libary convolve ') 

plt.subplot(2 , 1 , 2)
plt.plot(t1 , q)
plt.grid()
plt.ylabel('f1(t) ')
plt.xlabel ('Convolve by hand')
####################################################################

####################################################################
r = H2(t1)
plt.figure( figsize = (10, 7) )
plt.subplot(2 , 1 , 1)
plt.plot(t2 , conv(f2(t1),y))
plt.grid()
plt.ylabel('f2(t)')
plt.title('Libary convolve ') 

plt.subplot(2 , 1 , 2)
plt.plot(t1 , r)
plt.grid()
plt.ylabel('f2(t)')
plt.xlabel ('Convolve by hand')
####################################################################

####################################################################
u = H3(t1)
plt.figure( figsize = (10, 7) )
plt.subplot(2 , 1 , 1)
plt.plot(t2 , conv(f3(t1),y))
plt.grid()
plt.ylabel('f3(t)')
plt.title('Libary convolve ') 

plt.subplot(2 , 1 , 2)
plt.plot(t1 , u)
plt.grid()
plt.ylabel('f3(t)')
plt.xlabel ('Convolve by hand')
####################################################################
