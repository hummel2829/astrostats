
import matplotlib.pyplot as plt
import numpy as np
from math import factorial


p = 0.4
N =100
experiments = 10000
success = [*range(0,100)]
psuc = []
nsuc = []



Nsuccess = np.random.binomial(N, p, experiments)
Psuccess = Nsuccess/N
w = np.ones_like(Nsuccess)/len(Nsuccess)
exphist = np.histogram(Nsuccess,bins =experiments, weights = w)


def PB(k):
    return((factorial(N)/(factorial(k)*factorial(N-k)))*(p**k)*(1-p)**(N-k))

result = map(PB,success)
theoPB = list(result)

pchangedata = np.zeros([10000,])
for a in range(0,experiments):
    
    data = [np.random.binomial(1, p, 1)]
    for i in range(1,N):
        data = np.append(data,np.random.binomial(1, p, 1))
        
        if data[i]==data[i-1]==1:
            p = 0.3
        else:
            p = 0.4
    pchangedata[a] = np.sum(data)

pchangehist = np.histogram(pchangedata, bins=experiments,weights=w)


print(pchangedata)




# plot for P(k) and simulation with p=0.4 constant

figure1, axis = plt.subplots(1, 1,constrained_layout=True)


hist = axis.hist(Nsuccess,bins = 40, weights = w, color = 'w', edgecolor = 'b' , label='Sim')
axis.plot(success[20:60],theoPB[20:60], c='r', marker="o", label='PB(k)')
hist2 = axis.hist(pchangedata,bins = 40, weights = w, color = 'g', edgecolor = 'b' , label='Sim2')


font = {'fontname' : 'Times New Roman' , 'size' : 20}
axis.legend(loc='upper right',fontsize = 20)
#axis.set_title('avg = %1.3f' %avg + ' , var = %1.3f' %stdN ,**font)

#axis.set_title('Solar flares over 1000 days',**font)
axis.set_xlabel('successes',**font)
axis.set_ylabel('P(k)',**font)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

print(plt.axis()[0])

ylines = np.arange(0,0.1,0.02)
plt.hlines(ylines,plt.axis()[0],plt.axis()[1])

plt.grid()
plt.show()


