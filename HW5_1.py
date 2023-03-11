

import random
import matplotlib.pyplot as plt
import numpy as np
from math import factorial


p = 0.4
N =100
experiments = 10000
success = [*range(0,100)]
psuc = []
nsuc = []

########################use np binomial##################
########################coin flip out of loop#########

Nsuccess = np.random.binomial(N, p, experiments)
Psuccess = Nsuccess/N
w = np.ones_like(Nsuccess)/len(Nsuccess)
exphist = np.histogram(Nsuccess,bins =experiments, weights = w)


def PB(k):
    return((factorial(N)/(factorial(k)*factorial(N-k)))*(p**k)*(1-p)**(N-k))

result = map(PB,success)
theoPB = list(result)





figure, axis = plt.subplots(1, 1,constrained_layout=True)


#hist = axis.hist(Nsuccess,bins = 10, weights = w, color = 'b', edgecolor = 'b' , label='Sim')
#axis.scatter(success[20:60],theoPB[20:60], s=20, c='r', marker="o", label='PB(k)')
axis.plot(success[20:60],theoPB[20:60], c='r', marker="o", label='PB(k)')

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 20}
#axis.set_title('avg = %1.3f' %avg + ' , var = %1.3f' %stdN ,**font)

#axis.set_title('Solar flares over 1000 days',**font)
axis.set_xlabel('successes',**font)
axis.set_ylabel('P',**font)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)



plt.show()


