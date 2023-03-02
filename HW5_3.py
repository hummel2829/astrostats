

import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from math import exp
from math import sqrt


mean = 10
std = 1
n = 10
experiments = 10000

G10 = np.random.normal(mean,std,n)
meanG10 = np.mean(G10)


interval74 = [(meanG10 + (1.96*std/sqrt(n))) , (meanG10 - (1.96*std/sqrt(n)))]


stdsample = np.std(G10)
df = n-1
t = 2.262 #from t critical value table in textbook

interval715 = [(meanG10 + (t*stdsample/sqrt(n))) , (meanG10 - (t*stdsample/sqrt(n)))]

print(interval74 , interval715)


count = 0
mean3 = []


for i in range(0,experiments):
    G10 = np.random.normal(mean,std,n)
    mean3.append(np.mean(G10))
    UB74 = np.mean(G10) + (1.96*std/sqrt(n))
    LB74 = np.mean(G10) - (1.96*std/sqrt(n))
    if LB74 <= mean <= UB74:
        count = count + 1

fracininterval = count/experiments
print(fracininterval)

#mean3prob = [x/sum(mean3count) for x in mean3count]

mean4 = []

for i in range(0,experiments):
    G10 = np.random.normal(mean,std,n)
    mean4.append(np.mean(G10))
    UB = (np.mean(G10) + (t*np.std(G10)/sqrt(n)))
    LB = (np.mean(G10) + (t*np.std(G10)/sqrt(n)))
    if LB <= mean <= UB:
        count = count + 1

fracininterval = count/experiments
print(fracininterval)




figure, axis = plt.subplots(1, 1,constrained_layout=True)

axis.hist(mean3, bins = 100 ,stacked = True , density = True)

#axis.bar(mean3,mean3prob, color = 'w', edgecolor = 'b' , width = 0.4, label='Sim')
#axis.scatter(success[35:45],theoPB[35:45], s=20, c='r', marker="o", label='PB(k)')

#axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 20}
axis.set_title('mean74 = %1.2f' %meanG10 + ', mean715 %1.2f' %meanG10  ,**font)



#axis.set_title('Solar flares over 1000 days',**font)
axis.set_xlabel('successes',**font)
axis.set_ylabel('P',**font)

txt = 'CI74 = (%1.2f' %interval74[1] + ', %1.2f)' %interval74[0] + '   CI715 = (%1.2f' %interval715[1] + ', %1.2f)' %interval715[0]
plt.figtext(0.5, -0.1, txt, wrap=True, horizontalalignment='center', fontsize=12)

plt.show()









