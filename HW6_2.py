
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

flaresdayarray = np.copy(flares)
flaresdayarray.resize(ceil(minfrom1stflare[-1]/mininday),mininday)

flaressuminday = np.sum(flaresdayarray,axis=1)

flaresweekarray = np.copy(flaresdayarray)
mininweek = 7*mininday
flaresweekarray.resize(ceil(minfrom1stflare[-1]/mininweek),mininweek)

flaressuminweek = np.sum(flaresweekarray,axis=1)


flares2weekarray = np.copy(flaresdayarray)
minin2week = 14*mininday
flares2weekarray.resize(ceil(minfrom1stflare[-1]/minin2week),minin2week)

flaresavgin2week = np.average(flares2weekarray,axis=1)





monthinyr = 31
flaresinmonth = np.copy(flaressuminday)
flaresinmonth.resize(ceil(len(flaressuminday)/monthinyr),monthinyr)
flaresmonthsum = np.sum(flaresinmonth,axis=1)
flaresavginmonth = np.average(flaresinmonth, axis =1)

months = np.arange(0,flaresavginmonth.shape[0])

a,b = np.polyfit(months, flaresavginmonth,1)


figure, axis = plt.subplots(1, 1,constrained_layout=True)

axis.scatter(months, flaresavginmonth, s=20, c='r', marker="o", label='exp(k)')
axis.plot(months, months*a +b)
#axis.plot(flaresavginmonth, 'or')
#axis.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
#axis.plot(flaressuminday, 'or')
#axis.plot(flaresavgin2week, 'or')
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


















