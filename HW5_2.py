

import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from math import exp



################### use np array of 1s and 0s NOT list 




filename = 'xray.txt'
xraydata = np.loadtxt(filename)
xraydata = xraydata.astype(int)

rows =[*range(0,xraydata.shape[0])]

dates = [datetime.datetime(*x) for x in xraydata]
successes = [*range(0,len(dates))]


cc = np.diff(dates)
timediff = [(x.seconds) for x in cc]
timediffhours = [x/3600 for x in timediff]
timediffcount = [timediffhours.count(x) for x in timediffhours]
totalflarecount = len(timediff)



y = totalflarecount/(sum(timediffhours))

#def expdist(x):
#    y*exp(-y*x)

timediffprob = [x/(sum(timediff)/(24*3600)) for x in timediffhours ]

expdist = [y*exp(-y*x) for x in successes]

secondinday = 3600*24
#Pp = [(((y*secondinday)**k)*(exp(-y*secondinday))/factorial(k)) for k in successes[0:100] ]

PB = []





figure, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(timediffcount,timediffprob, color = 'b', edgecolor = 'b' , width = 0.4, label='Sim')
axis.scatter(successes[0:100],expdist[0:100], s=20, c='r', marker="o", label='exp(k)')
#axis.scatter(successes[0:100],Pp, s=20, c='g', marker="o", label='Pp(k)')


axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 20}
#axis.set_title('avg = %1.3f' %avg + ' , var = %1.3f' %stdN ,**font)

#axis.set_title('Solar flares over 1000 days',**font)
axis.set_xlabel('time intervals in hours',**font)
axis.set_ylabel('P',**font)


plt.show()




