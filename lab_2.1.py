#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:16:52 2021

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
t1 = np.arange(0 , 10 + steps , steps ) # Add a step size to make sure the
                                         # plot includes 5.0. Since np. arange () only
                                         # goes up to , but doesn ’t include the
                                         # value of the second argument

print('Number of elements : len(t) = ', len( t1 ) , '\nFirst Element : t[0] = ', t1 [0] ,
      '\nLast Element : t[len(t) - 1] = ', t1 [len( t1 ) - 1])


# Notice the array might be a different size than expected since Python starts
# at 0. Then we will use our knowledge of indexing to have Python print the
# first and last index of the array . Notice the array goes from 0 to len () - 1    

# --- User - Defined Function ---

# Create output y(t) using a for loop and if/ else statements
def func1( t1 ) : # The only variable sent to the function is t
    y = np.zeros( t1.shape ) # initialze y(t) as an array of zeros  

    for i in range(len( t1 ) ) : # run the loop once for each index of t                                   
    
            y[ i ] = np.cos(t1[ i ])
    return y # send back the output stored in an array


y = func1( t1 ) # call the function we just created
plt.figure( figsize = (10, 10) )
plt.subplot(2 , 1 , 1)
plt.plot(t1 , y )
plt.grid()
plt.ylabel('y(t)  ')
plt.title('cosine Wave!!!! ')

#t = np.arange(0 , 10 + 0.25 , 0.25) # redefine t with poor resolution
#y = func1( t )

#plt.subplot(2 , 1 , 2)
#plt.plot(t , y )
#plt.grid()
#plt.ylabel('y(t) with Poor Resolution ')
#plt.xlabel ('t')
#plt.show()



