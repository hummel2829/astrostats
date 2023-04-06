
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
intervalnumber = np.arange(0,len(minbetween))

avgtimeinterval = np.mean(minbetween) # avg time between flares or lam for poisson dist
vartimeinterval = np.std(minbetween)**2 # variance of time between


intervalbootmean = np.average(np.random.choice(minbetween,[10000,20]),axis=1)

#Hnull is intervalbootmean == vartimeinterval
#Halt is intervalbootmean < vartimeinterval
#alpha = 0.01, Z table value 2.33, significance level 0.01

zinterval = (np.mean(intervalbootmean) - np.std(minbetween)**2)/(np.std(minbetween)/(10000**0.5))

# zinterval = -23104 < 2.33 so reject Hnull




#intervalover400index = np.argwhere(minbetween>400).T
intervalover400index = np.transpose(np.nonzero(minbetween>400))
xforintervalover400 = np.arange(len(minbetween))[[intervalover400index]]

m,b = np.polyfit(xforintervalover400.T[0], minbetween[[intervalover400index]], 1)

movingprod = np.zeros(len(minbetween))
for i in range (0,len(minbetween)-1):
    movingprod[i] = minbetween[i]*minbetween[i+1]
    
pearcorrcoff = np.sum(movingprod)/(len(minbetween)*np.std(minbetween)**2)

# ~0.9 so strong correlation from one interval to next

movingproddiff = np.diff(movingprod)
intervaldiff = np.diff(minbetween)




figure1, axis = plt.subplots(1, 1,constrained_layout=True)

x = np.arange(0,np.max(minbetween))
#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
axis.scatter(x,minbhist[0], s=50, c='r', marker="o", label='interval count')


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

axis.set_title('avg interval = %1.3f' %np.mean(minbetween) + ' , std interval = %1.3f' %np.std(minbetween) + ' , z score = %1.5f < 2.33' %zinterval ,**font)
axis.set_ylabel('counts',**font)
axis.set_xlabel('minutes between flares',**font)




figure2, axis = plt.subplots(1, 1,constrained_layout=True)

#x = np.arange(0,np.max(minbetween))
#axis.scatter(x,minbhist[0], s=100, c='r', marker="o", label='poisson light')
axis.scatter(intervalnumber,minbetween, c='b', marker="o", label='interval < 400min')
axis.scatter(xforintervalover400,minbetween[[intervalover400index]], c='r', marker="o", label='intervals > 400min')
axis.plot(xforintervalover400.T[0],m*xforintervalover400.T[0] + b, c='r', marker="o", label='y = 0.006x + 647.87')

axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg intercept = %1.3f' %avga5 + ' , std intercept = %1.3f' %stda5 ,**font)
axis.set_ylabel('time interval',**font)
axis.set_xlabel('number of intervals',**font)





figure3, axis = plt.subplots(1, 1,constrained_layout=True)

#x = np.arange(0,np.max(minbetween))
#axis.scatter(x,minbhist[0], s=100, c='r', marker="o", label='poisson light')
xx = np.arange(0,len(movingprod)-1)
axis.scatter(xx,movingproddiff, c='b', marker="o", label='min btwn flares')

#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

axis.set_title('avg y = %1.3f' %np.mean(movingproddiff) + ' , std y = %1.3f' %np.std(movingproddiff) ,**font)
axis.set_ylabel('y interval[t+3]*interval[t+2] - interval[t+1]*interval[t] ',**font)
axis.set_xlabel('number of intervals',**font)




figure4, axis = plt.subplots(1, 1,constrained_layout=True)

#x = np.arange(0,np.max(minbetween))
#axis.scatter(x,minbhist[0], s=100, c='r', marker="o", label='poisson light')
xx = np.arange(0,len(intervaldiff))
axis.scatter(xx,intervaldiff, c='b', marker="o", label='min btwn flares')

#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

axis.set_title('avg intercept diff = %1.3f' %np.mean(intervaldiff) + ' , std intercept = %1.3f' %np.mean(intervaldiff) ,**font)
axis.set_ylabel('interval[t+1] - interval[t]',**font)
axis.set_xlabel('number of interval diff',**font)




figure5, axis = plt.subplots(1, 1,constrained_layout=True)

axis.plot(movingprod, c='r', marker="o", label='interval product')


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg intercept = %1.3f' %pearcorrcoff + ' , std intercept = %1.3f' %stda5 ,**font)
axis.set_title('pearson corr coff = %1.3f' %pearcorrcoff ,**font)
axis.set_ylabel('interval[t]*interval[t+1]',**font)
axis.set_xlabel('number of products',**font)




plt.grid()
plt.show()




