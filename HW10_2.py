
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from math import floor

#look up np.allclose


freq = 40

############# from HW9

N = 1000


q = N/2 #N is even so N=2q
i = np.arange(0,q)
f = i/N
#f = np.arange(0,q/1000,1/1000)

# N is even so bq = 0
a0 = np.zeros(int(q))
a = np.zeros(int(q))
b = np.zeros(int(q))

for j in range(0,len(i)):
    s = 0
    c = 0
    a0sum = 0
    for t in range(0,N):
        signal = 2*np.sin(2*np.pi*freq*t)
        c = c + signal*np.cos(2*np.pi*f[j]*t)
        s = s + signal*np.sin(2*np.pi*f[j]*t)
        
        a0sum = a0sum + signal

        
    a0[j] = a0sum/N
    a[j] = (2/N)*c
    b[j] = (2/N)*s

I_f = (N/2)*((a)**2 + b**2)


######################################
'''

samplerate = 10000
t = np.arange(0,1,(1/samplerate))
pad = 0

y = 2*np.sin(2*np.pi*freq*t)
y = np.append(y,np.zeros(pad))
#pad either side doesn't matter


fftsin = fft(y)
N = len(fftsin)
n = np.arange(len(fftsin))
T = N/samplerate
freq = n/T



figure1, axis = plt.subplots(2, 1,constrained_layout=True)



t = np.arange(0,1,1/(samplerate+pad))

axis[0].scatter(t, y, s=50, c='r', marker="o", label='interval count')
axis[1].scatter(freq, np.abs(fftsin), s=50, c='r', marker="o", label='interval count')
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

'''

figure2, axis = plt.subplots(1, 1,constrained_layout=True)

axis.scatter( f*100, I_f, s=50, c='r', marker="o")


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

axis.set_title('Fourier spectrum HW9',**font)
axis.set_xlabel('frequency (Hz)',**font)
axis.set_ylabel('Intensity(Arb. Units)',**font)




plt.grid()
plt.show()



