#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 07:29:11 2021

@author: ChristopherThomasOsborne
"""
#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_11                                                        #
# 11-9-21                                                       #
#                                                               #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from matplotlib import patches  

# -- Zplane Code Given:    
def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()

    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    return z, p, k

# -- Defined H(z) = Y(z)/X(z):
num = [2, -40, 0]
den = [1, -10, 16]

# -- Use scipy.signal.residuez() to verify your partial fraction expansion:
root, pole, _ = sig.residuez(num, den) 
print('Find the Roots and Poles for H(z):')
print('Root = ', root)
print( 'Pole = ', pole)
print('\n')

# -- print out the values from the zplane() function:
z, p, k = zplane(num, den)
print('Values from the zplane()')
print('zero = ', z)
print('pole = ', p)
print('Gain (k) = ', k)

# -- Plots:
frequency, phase = sig.freqz(num, den, whole = True)

w = frequency/(2*np.pi)
mag_dB = 20*np.log10(phase)
angle_degree = np.angle(phase)*180/np.pi

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(w, mag_dB)
plt.title('mag_dB vs. W rad/s')
plt.grid()
plt.ylabel('mag_dB')


plt.subplot(2,1,2)
plt.title('Phase_angle vs. W rad/s')
plt.plot(w, angle_degree)
plt.grid()
plt.ylabel('Phase_angle')
plt.show()