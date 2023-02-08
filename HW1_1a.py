import numpy as np
import matplotlib.pyplot as plt

X = []

n = 100
draws = 10000

for i in range(0,10000):
    #print(i)
    C = np.random.randn(1,draws)
    Xbar = np.mean(C)
    X = np.append(X,Xbar)
    
plt.hist(X)
