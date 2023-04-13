
import random
import matplotlib.pyplot as plt
import numpy as np
from math import pi


N = 1000


q = N/2 #N is even so N=2q
i = np.arange(0,q)
f = i/N

# N is even so bq = 0
a0 = np.zeros(int(q))
a = np.zeros(int(q))
b = np.zeros(int(q))

for j in range(0,len(i)):
    s = 0
    c = 0
    a0sum = 0
    for t in range(0,N):
        ywhite = np.random.normal(loc = 0, scale = 1, size = 1)
        c = c + ywhite*np.cos(2*pi*f[j]*t)
        s = s + ywhite*np.sin(2*pi*f[j]*t)
        a0sum = a0sum + ywhite

        
    a0[j] = a0sum/N
    a[j] = (2/N)*c
    b[j] = (2/N)*s

I_f = (N/2)*((a0+a)**2 + b**2)



#### parametric bootstrap
sample = 100

aboot = np.zeros(sample)
bboot = np.zeros(sample)
a0boot = np.zeros(sample)

for h in range(0,sample):
    j = 1
    c = 0
    s = 0
    a0sum = 0
    for t in range(0,N):
        ywhite = np.random.normal(loc = 0, scale = 1, size = 1)
        c = c + ywhite*np.cos(2*pi*f[j]*t)
        s = s + ywhite*np.sin(2*pi*f[j]*t)
        a0sum = a0sum + ywhite
    aboot[h] = c*2/N
    bboot[h] = s*2/N
    a0boot[h] = a0sum/N

If2sample =  (N/2)*((a0boot+aboot)**2 + bboot**2)



If2bootdist = np.random.normal(loc = np.mean(If2sample), scale = np.std(If2sample),size = 10000)

If2bootsample = np.random.choice(If2bootdist,size = [10,1000])
If2bootsamplemean = np.average(If2bootsample,axis = 0)

t = 3.25 #alpha = 0.01 and degrees of freedom = sample-1 = 10, from table in Devore
UB = (np.mean(If2bootsample) + (t*np.std(If2bootsample)/(1000**0.5)))
LB = (np.mean(If2bootsample) - (t*np.std(If2bootsample)/(1000**0.5)))


figure1, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
axis.scatter( f, I_f, s=50, c='r', marker="o", label='interval count')


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

axis.set_title('Fourier spectrum of white noise',**font)
axis.set_ylabel('frequency (Hz)',**font)
axis.set_xlabel('Intensity(Arb. Units)',**font)


figure2, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
axis.hist(If2bootdist,bins=50)

#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg interval = %1.3f' %np.mean(minbetween) + ' , std interval = %1.3f' %np.std(minbetween) + ' , z score = %1.5f < 2.33' %zinterval ,**font)
axis.set_ylabel('counts',**font)
axis.set_xlabel('I_f2 values',**font)




plt.grid()
plt.show()









