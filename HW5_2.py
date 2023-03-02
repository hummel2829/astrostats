

import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from math import exp


filename = 'xray.txt'
xraydata = np.loadtxt(filename)
xraydata = xraydata.astype(int)

rows =[*range(0,xraydata.shape[0])]

dates = [datetime.datetime(*x) for x in xraydata]
successes = [*range(0,len(dates))]

print(dates[1]-dates[0])

cc = np.diff(dates)
timediff = [(x.seconds) for x in cc]
timediffhours = [x/3600 for x in timediff]
timediffcount = [timediff.count(x) for x in timediff ]


timediffprob = [x/sum(timediffcount) for x in timediffcount ]

y = sum(timediffcount)/(sum(timediff))

#def expdist(x):
#    y*exp(-y*x)

expdist = [y*exp(-y*x) for x in successes]







figure, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(timediffhours,timediffprob, color = 'b', edgecolor = 'b' , width = 0.4, label='Sim')
#axis.scatter(successes,expdist, s=20, c='r', marker="o", label='PB(k)')

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 20}
#axis.set_title('avg = %1.3f' %avg + ' , var = %1.3f' %stdN ,**font)

#axis.set_title('Solar flares over 1000 days',**font)
axis.set_xlabel('time intervals in hours',**font)
axis.set_ylabel('P',**font)


plt.show()




