


import numpy as np
import matplotlib.pyplot as plt


n = 100
draws = 10000

X=[]
mu, sigma = 72,3

for i in range(0,draws):
    #print(i)
    C =  np.random.normal(mu,sigma,n)
    Xbar = np.mean(C)
    X = np.append(X,Xbar)




ER=[]

for i in range(0,draws):
    if X[i] <= (np.mean(X)+1) and X[i] >=(np.mean(X)-1):
        ER=np.append(ER,X[i])

print(len(ER)/draws)





