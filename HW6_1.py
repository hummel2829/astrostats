import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


N = 1000
NB = 10000
n = 20
alpha = 1
beta = 1
sigma = 0.2

eps = np.random.normal(0,sigma,N)
x = np.arange(0,N)/N
y = beta*x + alpha + eps

pairs = np.array([x,y]).T

rng = np.random.default_rng()
n20 = rng.choice(pairs,n)

#HW6_1 question 3
x3,y3 = n20[:,0], n20[:,1]

b3 = np.sum((x3-np.average(x3))*(y3-np.average(y3)))/np.sum((x3-np.average(x3))**2)

a3 = np.average(y3) - b3*np.average(x3)

print('HW6_1_3:a,b',a3,b3)


#HW6_1 question 4
a4,b4 = np.polyfit(n20[:,0], n20[:,1],1)
print('HW6_1_4:a,b',a4,b4)


#HW6_1 question 5

rng = np.random.default_rng()
nbootraw = rng.choice(pairs,(n*NB))

nboot = np.reshape(nbootraw,[NB,n,2])
a5 = []
b5 = []

for i in range(0,NB):
    a, b = np.polyfit(nboot[i,:,0],nboot[i,:,1],1)
    a5 = np.append(a5,a)
    b5 = np.append(b5,b)

avga5 = np.average(a5)
stda5 = np.std(a5)
avgb5 = np.average(b5)
stdb5 = np.std(b5)


#95% interval calculation

bins = 100
a5hist = np.histogram(a5,bins)

for i in range(0,bins):
    percent = np.sum(a5hist[0][i:-(i+1)]/10000)
    if percent <= 0.95:
        print('HW6_1_5   95% con interval for a: ', a5hist[1][i],a5hist[1][-i+1])
        break
    

b5hist = np.histogram(b5,bins)

for i in range(0,bins):
    percent = np.sum(b5hist[0][i:-(i+1)]/10000)
    if percent <= 0.95:
        print('HW6_1_5   95% con interval for b: ',b5hist[1][i],b5hist[1][-i+1])
        break
    






figure1, axis = plt.subplots(1, 1,constrained_layout=True)

axis.scatter(x, y, s=20, c='r', marker="o", label='Population')
axis.scatter(n20[:,0], n20[:,1], s=100, c='b', marker="o", label='sample')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
axis.legend(loc='upper left',fontsize=20)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)


axis.set_title('Population and sample of y=1*x + 1',**font)
axis.set_xlabel('x-values',**font)
axis.set_ylabel('y-values',**font)




figure2, axis = plt.subplots(1, 1,constrained_layout=True)

plt.hist(a5)

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)


axis.set_title('avg intercept = %1.3f' %avga5 + ' , std intercept = %1.3f' %stda5 ,**font)
axis.set_xlabel('intercept values',**font)
axis.set_ylabel('number of intercept values',**font)



figure3, axis = plt.subplots(1, 1,constrained_layout=True)

plt.hist(b5)

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)


axis.set_title('avg slope = %1.3f' %avgb5 + ' , std slope = %1.3f' %stdb5 ,**font)
axis.set_xlabel('slope values',**font)
axis.set_ylabel('number of slope values',**font)



plt.grid()
plt.show()



