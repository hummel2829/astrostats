
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

allminutes = np.zeros(minfrom1stflare[-1]+1)

np.put(allminutes, minfrom1stflare, 1)

flares = allminutes.astype(int)

mininhour = int(60)

flareshourarray = np.copy(flares)
flareshourarray.resize(ceil(minfrom1stflare[-1]/mininhour),mininhour)

flaressuminhour = np.sum(flareshourarray,axis=1)

fgivenf = 0
fgivennf = 0
nfgivenf = 0
nfgivennf = 0

for i in range(0,len(flaressuminhour)-1):
    if flaressuminhour[i] > 0 and flaressuminhour[i+1] > 0:
        fgivenf = fgivenf + 1
    if flaressuminhour[i] == 0 and flaressuminhour[i+1] > 0:
        fgivennf = fgivennf + 1
    if flaressuminhour[i] > 0 and flaressuminhour[i+1] == 0:
        nfgivenf = nfgivenf + 1
    if flaressuminhour[i] == 0 and flaressuminhour[i+1] == 0:
        nfgivennf = nfgivennf + 1

Pfgivenf = fgivenf/(len(flaressuminhour)-1)
Pfgivennf = fgivennf/(len(flaressuminhour)-1)
Pnfgivenf = nfgivenf/(len(flaressuminhour)-1)
Pnfgivennf = nfgivennf/(len(flaressuminhour)-1) 





mininday = int(24*60)

flaresdayarray = np.copy(flares)
flaresdayarray.resize(ceil(minfrom1stflare[-1]/mininday),mininday)

flaressuminday = np.sum(flaresdayarray,axis=1)

flaresweekarray = np.copy(flaresdayarray)
mininweek = 7*mininday
flaresweekarray.resize(ceil(minfrom1stflare[-1]/mininweek),mininweek)

#flaressuminweek = np.sum(flaresweekarray,axis=1)


#flares2weekarray = np.copy(flaresdayarray)
#minin2week = 14*mininday
#flares2weekarray.resize(ceil(minfrom1stflare[-1]/minin2week),minin2week)

#flaresavgin2week = np.average(flares2weekarray,axis=1)



monthinyr = 31
flaresinmonth = np.copy(flaressuminday)
flaresinmonth.resize(ceil(len(flaressuminday)/monthinyr),monthinyr)
flaresmonthsum = np.sum(flaresinmonth,axis=1)
flaresavginmonth = np.average(flaresinmonth, axis =1)

months = np.arange(0,flaresavginmonth.shape[0])


#avgfirst10 = np.average(flaresavginmonth[0:10])
#stdfirst10 = np.std(flaresavginmonth[0:10])

#avglast10 = np.average(flaresavginmonth[-1:(months[-1]-11):-1])
#stdlast10 = np.std(flaresavginmonth[-1:(months[-1]-11):-1])

a,b = np.polyfit(months, flaresavginmonth,1)


figure1, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.scatter(months, flaresavginmonth, s=20, c='r', marker="o", label='exp(k)')
#axis.plot(months, months*a +b)
axis.plot(flaresmonthsum/31)
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


#average flare/day by month

figure2, axis = plt.subplots(1, 1,constrained_layout=True)

axis.scatter(months, flaresmonthsum/31, s=20, c='r', marker="o", label='exp(k)')
axis.plot(months, months*a +b)

axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.set_title('avg intercept = %1.3f' %avga5 + ' , std intercept = %1.3f' %stda5 ,**font)
axis.set_xlabel('avg flares/day',**font)
axis.set_ylabel('month',**font)

plt.grid()
plt.show()



















