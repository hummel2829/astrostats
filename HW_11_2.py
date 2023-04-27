
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft


y = np.array([0,1,-1,0])
N = len(y)
i = np.arange(N)+1
f = (i/N)
fcrossi = np.outer(f,i)

a0 = (1/N)*np.sum(y)

acomp = y*np.cos(2*np.pi*fcrossi)
a = (2/N)*np.sum(acomp,axis=1)

bcomp = y*np.sin(2*np.pi*fcrossi)
b = (2/N)*np.sum(bcomp,axis=1)

k = np.arange(N)
t = np.arange(N)
kcrosst = np.outer(k,t)
c = np.sum(y*np.exp(-2*np.pi*1j*kcrosst/N), axis = 1)








#def fftcoef(y):
#    c = fft(y)