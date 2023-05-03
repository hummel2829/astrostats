
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import seaborn

N = 1000

T1 = 20
f1 = 1/T1
t1 = np.arange(N)

y1 = np.sin(2*np.pi*f1*t1)


T2 = 10
f2 = 1/T2
t2 = np.arange(N,2*N)

y2 = np.sin(2*np.pi*f2*t2)

signal = np.append(y1,y2)



t = np.arange(0,1,0.001)
f = 40
ytest = 3*np.sin(2*np.pi*f*t) + np.sin(2*np.pi*0.25*f*t)


def fftcoef(y):
    N = len(y)
    if N%2 == 0:
        f = np.arange(N/2)/N
    else:
        f = np.arange((N-1)/2)/N
    c = fft(y)
    a = np.real(c)
    b = np.imag(c)
    return  (c, a, b, f, N) 

Nsignal = fftcoef(signal)[4]
csignal = fftcoef(signal)[0]
asignal = (2/N)*fftcoef(signal)[1]
bsignal = (2/N)*fftcoef(signal)[2]
fsignal = fftcoef(signal)[3]

Ny1 = N
cy1 = fftcoef(y1)[0]
ay1 = (2/N)*fftcoef(y1)[1]
by1 = (2/N)*fftcoef(y1)[2]
fy1 = fftcoef(y1)[3]

Ny2 = N
cy2 = fftcoef(y2)[0]
ay2 = (2/N)*fftcoef(y2)[1]
by2 = (2/N)*fftcoef(y2)[2]
fy2 = fftcoef(y2)[3]


ccombine = np.stack((cy1,cy2),axis=0)

signalheat = np.reshape(signal,(200,10))
cheat = np.zeros([signalheat.shape[0],signalheat.shape[1]], dtype = np.complex)

for i in range(0,len(signalheat)):
    cheat[i] = fftcoef(signalheat[i])[0]

intensity = np.real(cheat*np.conj(cheat))


'''

figure1, axis = plt.subplots(1, 1,constrained_layout=True)

#freq = np.arange(0,len(y))
t = np.arange(len(signal))

#axis.scatter(t, signal, s=50, c='r', marker="o", label='interval count')
axis.plot(t, signal, c='r', marker="o", label='interval count')

#plt.xlim(0,100)


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel('frequency (Hz)',**font)
axis.set_ylabel('Intensity(Arb. Units)',**font)




figure2, axis = plt.subplots(1, 1,constrained_layout=True)

freq = np.arange(0,len(signal))
t = np.arange(len(signal))

axis.scatter(freq, np.abs(bsignal), s=50, c='r', marker="o", label='interval count')
axis.scatter(freq, np.abs(asignal), s=50, c='r', marker="o", label='interval count')

plt.xlim(0,250)




#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel('frequency (Hz)',**font)
axis.set_ylabel('Intensity(Arb. Units)',**font)

'''


figure3, axis = plt.subplots(1, 1,constrained_layout=True)

freq = np.arange(0,len(y1))*1e-4
t = np.arange(len(y1))

axis.scatter(freq, np.abs(by1), s=50, c='r', marker="o", label='interval count')
axis.scatter(freq, np.abs(ay1), s=50, c='r', marker="o", label='interval count')

#plt.xlim(0,0.5)


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel('frequency (Hz)',**font)
axis.set_ylabel('Intensity(Arb. Units)',**font)


figure4, axis = plt.subplots(1, 1,constrained_layout=True)

freq = np.arange(0,len(y2))*1e-4
t = np.arange(len(y2))

axis.scatter(freq, np.abs(by2), s=50, c='r', marker="o", label='interval count')
axis.scatter(freq, np.abs(ay2), s=50, c='r', marker="o", label='interval count')

plt.xlim(0,0.5)


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel('frequency (Hz)',**font)
axis.set_ylabel('Intensity(Arb. Units)',**font)


figure5, axis = plt.subplots(1, 1,constrained_layout=True)

freq = np.arange(0,len(y2))*1e-4
t = np.arange(len(y2))


seaborn.heatmap(intensity)


#axis.scatter(freq, np.abs(by2), s=50, c='r', marker="o", label='interval count')
#axis.scatter(freq, np.abs(ay2), s=50, c='r', marker="o", label='interval count')

#plt.xlim(0,0.5)


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel('frequency (Hz)',**font)
axis.set_ylabel('Intensity(Arb. Units)',**font)



plt.grid()
plt.show()















