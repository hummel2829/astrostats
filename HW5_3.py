

import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from math import exp
from math import sqrt


mean = 10
std = 1
n = 10
experiments = 10000

G10 = np.random.normal(mean,std,n)
meanG10 = np.mean(G10)


interval74 = [(meanG10 + (1.96*std/sqrt(n))) , (meanG10 - (1.96*std/sqrt(n)))]


stdsample = np.std(G10)
df = n-1
t = 2.262 #from t critical value table in textbook

interval715 = [(meanG10 + (t*stdsample/sqrt(n))) , (meanG10 - (t*stdsample/sqrt(n)))]

print(interval74 , interval715)


count = 0

for i in range(0,experiments):
    G10 = np.random.normal(mean,std,n)
    UB74 = np.mean(G10) + (1.96*std/sqrt(n))
    LB74 = np.mean(G10) - (1.96*std/sqrt(n))
    if LB74 <= mean <= UB74:
        count = count + 1

fracininterval = count/experiments
print(fracininterval)


for i in range(0,experiments):
    G10 = np.random.normal(mean,std,n)
    UB = (np.mean(G10) + (t*np.std(G10)/sqrt(n)))
    LB = (np.mean(G10) + (t*np.std(G10)/sqrt(n)))
    if LB <= mean <= UB:
        count = count + 1

fracininterval = count/experiments
print(fracininterval)











