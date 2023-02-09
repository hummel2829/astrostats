# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:38:44 2023

@author: John
"""

import numpy as np
import matplotlib.pyplot as plt

redballs = np.array([1,1,1,1])
blueballs = np.array([2,2,2,2,2])
greenballs = np.array([3,3,3,3,3,3])

balls =np.concatenate([redballs,blueballs,greenballs])

N = 5000

selections = np.empty([N,3])

color = 1

count = 0
for i in range(0,N):
    selections = np.random.choice(balls,3)
    if selections[0]==selections[1]==color or selections[1]==selections[2]==color or selections[2]==selections[0]==color:
        count = count +1
        print(selections)
    
print(count/N)
    
    
    
    
    





