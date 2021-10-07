#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:19:29 2021

@author: ChristopherThomasOsborne
"""

#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_6                                                         #
# 10-5-21                                                       #
# Any other necessary information needed to navigate the file   #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams.update({'font.size': 14})
steps = 1e-5 # Define step size
t1 = np.arange(0 , 2 + steps , steps ) 
#print('Number of elements : len(t) = ', len( t1 ) , '\nFirst Element : t[0] = ', t1 [0] ,
 #     '\nLast Element : t[len(t) - 1] = ', t1 [len( t1 ) - 1])





# --- User - Defined Function ---
####################################################################
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
####################################################################





# ---------- pre_lab, h(t) calculation: --------------
####################################################################
def ht(t):
    y = ((1/2) - ((1/2)*np.exp(-4*t)) + np.exp(-6*t)) * step(t) 
    return y
####################################################################





# ---------- pre_lab, H(s) calculation: --------------
####################################################################
HS1 = ([1,6,12],[1,10,24])
tout , yout = sig.step (((HS1)) , T = t1)
####################################################################





# ---------- pre_lab, H(s) and h(t) Plots: --------------
####################################################################
plt.figure( figsize = (10, 7) )
plt.subplot(2 , 1 , 1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('H(s)')
plt.title('Libary Convolve') 

plt.subplot(2 , 1 , 2)
plt.plot( t1, ht(t1))
plt.grid()
plt.ylabel('h(t)')
plt.xlabel ('convolve by Hand')
####################################################################





# ------------- Y (s) = H(s)X(s) ---------------------------
###################################################################
# This will print out the roots 
# X(s) = 1/s

num2 = [0,1,6,12]
den2 = [1,10,24,0]
root = sig.residue(num2, den2)

r2,p2,_= sig.residue(num2,den2)

print("Roots/Poles: Part (1)")
print('------------------------')
print('r2=', r2, "\n" 'p2=', p2)

####################################################################





# ------------- Y (s) = H(s)X(s) ---------------------------
###################################################################
# X(s) = 1/s
t2 = np.arange(0, 4.5+steps, steps)

num3 = [0,0,0,0,0,0,25250]
den3 = [1,18,218,2036,9085,25250,0]

r3,p3,_ = sig.residue(num3,den3)

print('\n')
print("Roots/Poles: part (2)")
print('------------------------')
print('r3=', r3, "\n" 'p3=', p3)
print('\n')

# -------------- Cosine meathod: h(t), part 2 ----------------------
yt_cos = 0
for i in range(len(r3)):
    phase_k = np.angle(r3[i])
    mag_k = np.abs(r3[i])
    w = np.imag(p3[i])
    a = np.real(p3[i])   
    yt_cos += mag_k*np.exp(a*t2)*np.cos(w*t2 + phase_k)*step(t2)
print('mag_k =',mag_k)
print('phase_k =',phase_k)
print('\n')
####################################################################






#------------------------ H(s), part 2: ------------------------
####################################################################
num4 = [0,0,0,0,0,25250]
den4 = [1,18,218,2036,9085,25250]

tout, yout = sig.step((num4, den4), T = t2)
####################################################################






# ---------- H(s) and h(t) Plots: Part 2:  --------------
####################################################################
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t2, yt_cos)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('step response y(t)')


plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('step response H(s)')
plt.tight_layout()
plt.show()
####################################################################

