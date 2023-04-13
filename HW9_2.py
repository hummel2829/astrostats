
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

I_f = (N/2)*(a**2 + b**2)



#### parametric bootstrap

aboot = np.zeros(61)
bboot = np.zeros(61)

for h in range(0,61):
    j = 1
    c = 0
    s = 0
    for t in range(0,N):
        ywhite = np.random.normal(loc = 0, scale = 1, size = 1)
        c = c + ywhite*np.cos(2*pi*f[j]*t)
        s = s + ywhite*np.sin(2*pi*f[j]*t)
        a0sum = a0sum + ywhite
    aboot[h] = c*2/N
    bboot[h] = s*2/N

If2boot =  (N/2)*(aboot**2 + bboot**2)

sample = 10
If2sample = np.zeros(sample)
aboot = np.zeros(sample)
bboot = np.zeros(sample)

for h in range(0,sample):
    j = 1
    c = 0
    s = 0
    for t in range(0,N):
        ywhite = np.random.normal(loc = 0, scale = 1, size = 1)
        c = c + ywhite*np.cos(2*pi*f[j]*t)
        s = s + ywhite*np.sin(2*pi*f[j]*t)
        a0sum = a0sum + ywhite
    aboot[h] = c*2/N
    bboot[h] = s*2/N
    If2sample[h] = (N/2)*(aboot[h]**2 + bboot[h]**2)

If2boot = np.random.choice(If2sample, size = 10000)


t = 2.66 #alpha = 0.01 and degrees of freedom = 60, from table in Devore
UB = (np.mean(If2boot) + (t*np.std(If2boot)/(sample**0.5)))
LB = (np.mean(If2boot) - (t*np.std(If2boot)/(sample**0.5)))


figure1, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
axis.scatter( f, I_f, s=50, c='r', marker="o", label='interval count')


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg interval = %1.3f' %np.mean(minbetween) + ' , std interval = %1.3f' %np.std(minbetween) + ' , z score = %1.5f < 2.33' %zinterval ,**font)
axis.set_ylabel('counts',**font)
axis.set_xlabel('minutes between flares',**font)


figure2, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
axis.hist(If2boot,bins=5)

#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg interval = %1.3f' %np.mean(minbetween) + ' , std interval = %1.3f' %np.std(minbetween) + ' , z score = %1.5f < 2.33' %zinterval ,**font)
axis.set_ylabel('counts',**font)
axis.set_xlabel('minutes between flares',**font)




plt.grid()
plt.show()









