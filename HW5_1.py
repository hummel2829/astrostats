

import random
import matplotlib.pyplot as plt
from math import factorial


p = 0.4
N =100
experiments = 10000
success = [*range(0,100)]
data = [1,0]
psuc = []
nsuc = []


for i in range (0,experiments):
    sample = random.choices(data,weights = [p,1-p], k=N)
    nsuc.append(sum(list(filter(lambda x: (x == 1), sample))))
    psuc.append(sum(list(filter(lambda x: (x == 1), sample)))/len(sample))

def prob(n):
    return nsuc.count(n)/experiments

result = map(prob,success)

Pdsuc = list(result)       #prob dist of success from sim

def number(n):
    return nsuc.count(n)

result = map(number,success)

ndsuc = list(result)


def PB(k):
    return((factorial(N)/(factorial(k)*factorial(N-k)))*(p**k)*(1-p)**(N-k))

result = map(PB,success)
theoPB = list(result)





figure, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(success[35:45],Pdsuc[35:45], color = 'w', edgecolor = 'b' , width = 0.4, label='Sim')
axis.scatter(success[35:45],theoPB[35:45], s=20, c='r', marker="o", label='PB(k)')

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 20}
#axis.set_title('avg = %1.3f' %avg + ' , var = %1.3f' %stdN ,**font)

#axis.set_title('Solar flares over 1000 days',**font)
axis.set_xlabel('successes',**font)
axis.set_ylabel('P',**font)


plt.show()


