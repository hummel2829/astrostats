# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:07:46 2023

@author: John
"""


import numpy as np
import matplotlib.pyplot as plt
import random


coin =[-1,1]
exp =10000
steps = 100

count=0
finalpos = []

for i in range(0,exp):
    trial = sum(random.choices(coin,k=steps))
    finalpos.append(trial)


plt.hist(finalpos,bins=40)

