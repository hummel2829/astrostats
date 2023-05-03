
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft


y = np.array([0,1,0,-1])
N = len(y)
i = np.arange(N)
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

t = np.arange(0,1,0.001)
f = 40
y = 3*np.sin(2*np.pi*f*t) + np.sin(2*np.pi*0.25*f*t)


def fftcoef(y):
    N = len(y)
    if N%2 == 0:
        f = np.arange(N/2)/N
    else:
        f = np.arange((N-1)/2)/N
    c = fft(y)
    a = np.real(c)
    b = np.imag(c)
    return  (c, a, b, f) 

    
cfun = fftcoef(y)[0]
afun = (2/N)*fftcoef(y)[1]
bfun = (2/N)*fftcoef(y)[2]
ffun = fftcoef(y)[3]


figure1, axis = plt.subplots(2, 1,constrained_layout=True)

freq = np.arange(0,len(y))

axis[0].scatter(t, y, s=50, c='r', marker="o", label='interval count')
axis[1].scatter(freq, np.abs(bfun), s=50, c='r', marker="o", label='interval count')
#m,stem,base = axis[1].stem(freq, np.abs(fftsin), label='interval count')
#stem.set_linewidth(10)
plt.xlim(0,100)




#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis[0].set_xlabel('frequency (Hz)',**font)
axis[0].set_ylabel('Intensity(Arb. Units)',**font)




plt.grid()
plt.show()














