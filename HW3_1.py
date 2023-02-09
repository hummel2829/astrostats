# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:41:56 2023

@author: John
"""


import numpy as np
import matplotlib.pyplot as plt
import random


visa = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
master = [2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
vm = [3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0]    #visa and master

sample =[1,1,1,1,1,3,3,3,3,3,2,2,2,0,0,0,0,0,0,0] # 1=visa , 2=master , 3=both , 0=neither




N = 10000           #number of experiments

counta = 0
countb = 0
countc = 0

for i in range(0,N):
    selections = np.random.choice(sample,1)
    if selections!=0:
        counta = counta +1
    if selections==0:
        countb = countb +1
    if selections==1:
        countc = countc +1
 

with open("output.txt",'w') as f:
    print('P of having visa or master',counta/N,
          '\nP of having neither',countb/N,
          '\nP of having visa only',countc/N,file=f)















    
