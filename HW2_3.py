# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:07:46 2023

@author: John
"""


import numpy as np
import matplotlib.pyplot as plt


coin =np.array([-1,1])
steps =10000

trip = np.zeros([steps,1],dtype=int)

for i in range(0,steps):
    trip[i] = np.random.choice(coin)
    
print(trip[0:9])


finish = 19
final = np.sum(trip[0:finish])
print(final)
