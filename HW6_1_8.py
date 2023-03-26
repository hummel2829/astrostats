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

#HW6_1 question 8

rng = np.random.default_rng()
nbootraw = rng.choice(pairs,(n*NB))

nboot = np.reshape(nbootraw,[NB,n,2])
a8 = []
b8 = []

for i in range(0,NB):
    a, b = np.polyfit(nboot[i,:,0],nboot[i,:,1],1)
    a8 = np.append(a8,a)
    b8 = np.append(b8,b)


ea = a8-alpha
eb = b8-beta


avgea = np.average(ea)
stdea = np.std(ea)
avgeb = np.average(eb)
stdeb = np.std(eb)

aevalues = []
for i in range(0,NB):
    ii = np.random.randint(0,ea.shape[0]-1,20)
    ae,be = np.polyfit(ea[ii], eb[ii],1)
    aevalues = np.append(aevalues,ae)
    
w = abs(aevalues/len(aevalues))

aebins = 100
aehist = np.histogram(aevalues,aebins)
print(ae)

figure1, axis = plt.subplots(1, 1,constrained_layout=True)
plt.hist(aevalues,bins=50, weights = w)
#axis.scatter(eb,ea, s=20, c='r', marker="o", label='exp(k)')
#axis.scatter(successes[0:100],Pp, s=20, c='g', marker="o", label='Pp(k)')


axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#axis.set_title('avg y-axis = %1.3f' %avgeb + ' , std y-axis = %1.3f' %stdeb + ',  avg y-axis = %1.3f' %avgeb + ' , std y-axis = %1.3f' %stdeb ,**font)


#axis.set_xlabel('(sample intercept)-(population intercept)',**font)
#axis.set_ylabel('(sample slope)-(population slope)',**font)


figure2, axis = plt.subplots(1, 1,constrained_layout=True)

axis.plot(eb[0:100]*ea[0:100],'ro')

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 20}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

axis.set_title('(a-alpha)*(b-beta)',**font)
axis.set_xlabel('(index)',**font)
axis.set_ylabel('(intensity)',**font)


plt.grid()
plt.show()














