
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
from math import ceil



filename = 'xray.txt'
xraydata = np.loadtxt(filename)
xraydata = xraydata.astype(int)

rows =[*range(0,xraydata.shape[0])]


dates = [datetime.datetime(*x) for x in xraydata]
successes = [*range(0,len(dates))]

#minutes from first flare
minfrom1stflare = [(((dates[x] - dates[0]).days)*24*60) + ((dates[x] - dates[0]).seconds)//60 for x in rows]
minfrom1stflare = np.asarray(minfrom1stflare)

minbetween = np.diff(minfrom1stflare)
minbhist = np.histogram(minbetween,bins=np.max(minbetween))

avgtimeinterval = np.mean(minbetween) # avg time between flares or lam for poisson dist
vartimeinterval = np.std(minbetween)**2 # variance of time between




'''

figure1, axis = plt.subplots(1, 1,constrained_layout=True)

x = np.arange(0,np.max(minbetween))
#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
axis.scatter(x,minbhist[0], s=100, c='r', marker="o", label='interval count')


axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg intercept = %1.3f' %avga5 + ' , std intercept = %1.3f' %stda5 ,**font)
axis.set_ylabel('counts',**font)
axis.set_xlabel('minutes between flares',**font)




figure2, axis = plt.subplots(1, 1,constrained_layout=True)

#x = np.arange(0,np.max(minbetween))
#axis.scatter(x,minbhist[0], s=100, c='r', marker="o", label='poisson light')
xx = np.arange(0,len(minbetween))
axis.scatter(xx,minbetween, c='b', marker="o", label='min btwn flares')

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg intercept = %1.3f' %avga5 + ' , std intercept = %1.3f' %stda5 ,**font)
axis.set_ylabel('time interval',**font)
axis.set_xlabel('number of intervals',**font)


'''


figure3, axis = plt.subplots(1, 1,constrained_layout=True)

#x = np.arange(0,np.max(minbetween))
#axis.scatter(x,minbhist[0], s=100, c='r', marker="o", label='poisson light')
xx = np.arange(0,len(rateofchangeinterval))
axis.scatter(xx,rateofchangeinterval, c='b', marker="o", label='min btwn flares')

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg intercept = %1.3f' %avga5 + ' , std intercept = %1.3f' %stda5 ,**font)
axis.set_ylabel('time interval',**font)
axis.set_xlabel('number of intervals',**font)




plt.grid()
plt.show()




