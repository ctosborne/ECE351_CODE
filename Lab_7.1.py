#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:35:58 2021

@author: ChristopherThomasOsborne
"""

#################################################################
#                                                               #
# Chris Osborne                                                 #
# ECE-351, Section-51                                           #
# Lab_7                                                         #
# 10-12-21                                                       #
# Any other necessary information needed to navigate the file   #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


# -----------  G(s) ------------------
G_num = [1,9]
G_den = sig.convolve([1,-6,-16],[1,4])
Z1, P1,_ = sig.tf2zpk(G_num, G_den)
print('Zeros of G(s): ', Z1, '  Poles of G(s): ', P1)

# -----------  B(s) ------------------
B_num = [1,26,168]
root = np.roots(B_num)
print('Zeros of B(s): ', root)


# ------------ A(s) ------------------
A_num = [1,4]
A_den = [1,4,3]
Z2, P2,_ = sig.tf2zpk(A_num, A_den)
print('Zeros of A(s): ', Z2, '  Poles of A(s): ', P2)


# ------------- A(s)*G(s) -------------
A_G_open_loop_num  = sig.convolve(A_num, G_num)
A_G_open_loop_den = sig.convolve(A_den, G_den)
print('A_G_open_loop_num: ', A_G_open_loop_num,'   A_G_open_loop_den: ', A_G_open_loop_den)


# -- Graphing the Open-Loop Transfer Function --
tout, yout = sig.step((A_G_open_loop_num, A_G_open_loop_den))
plt.figure(figsize = (10,7))
plt.plot(tout, yout)
plt.grid()
plt.title('Step Response of Open-Loop Transfer Function')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()


# -- Closed-Loop Transfer Function --
closed_loop_num = sig.convolve(A_num, G_num)
closed_loop_den = sig.convolve(G_den + sig.convolve(G_num, B_num), A_den)
Z3, P3,_ = sig.tf2zpk(closed_loop_num,closed_loop_den)
print('closed_loop_num: ', closed_loop_num, '   closed_loop_den: ', closed_loop_den)
print('closed-loop-num Zeros: ', Z3, '   closed-loop-den Poles: ', P3)


# -- Graphing the Closed-Loop Transfer Function --
tout, yout = sig.step((closed_loop_num, closed_loop_den))
plt.figure(figsize = (10,7))
plt.plot(tout, yout)
plt.grid()
plt.title('Step Response of Closed-Loop Transfer Function')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()