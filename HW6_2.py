
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from math import exp
from math import ceil



################### use np array of 1s and 0s NOT list 




filename = 'xray.txt'
xraydata = np.loadtxt(filename)
xraydata = xraydata.astype(int)

rows =[*range(0,xraydata.shape[0])]


dates = [datetime.datetime(*x) for x in xraydata]
successes = [*range(0,len(dates))]

#minutes from first flare
minfrom1stflare = [(((dates[x] - dates[0]).days)*24*60) + ((dates[x] - dates[0]).seconds)//60 for x in rows]

allminutes = np.zeros(minfrom1stflare[-1]+1)

np.put(allminutes, minfrom1stflare, 1)

flares = allminutes.astype(int)

mininday = int(24*60)

flares.resize(mininday,ceil(minfrom1stflare[-1]/mininday))

flaresrowsum = np.sum(flares,axis=1)




figure, axis = plt.subplots(1, 1,constrained_layout=True)

axis.plot(flaresrowsum, 'o')
#axis.bar(timediffcount,timediffprob, color = 'b', edgecolor = 'b' , width = 0.4, label='Sim')
#axis.scatter(flaresrowsum,, s=20, c='r', marker="o", label='exp(k)')
#axis.scatter(successes[0:100],Pp, s=20, c='g', marker="o", label='Pp(k)')


axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 20}
#axis.set_title('avg = %1.3f' %avg + ' , var = %1.3f' %stdN ,**font)

#axis.set_title('Solar flares over 1000 days',**font)
axis.set_xlabel('time intervals in hours',**font)
axis.set_ylabel('P',**font)

plt.show()




A = np.array([1,2,3,5])

A.resize(3,2)
print(A)

B = np.sum(A,axis=1)
print(B)


















