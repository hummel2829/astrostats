
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

'''

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
minbhist = np.histogram(minbetween,bins=100)

'''
photonrate = (1/3e8)
timeinterval = 1e8
lam = photonrate*timeinterval  #photons/nanosecond
pdata = np.random.poisson(lam,1000)
print(np.mean(pdata)**0.5 , np.std(pdata))


figure1, axis = plt.subplots(1, 1,constrained_layout=True)
axis.hist(pdata,bins=100)
#axis.hist(minbetween,bins=100)
#axis.plot(minbhist[0] , color="red")

#plt.hist(aevalues,bins=50, weights = w)
#axis.scatter(eb,ea, s=20, c='r', marker="o", label='exp(k)')
#axis.scatter(successes[0:100],Pp, s=20, c='g', marker="o", label='Pp(k)')


#axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

#axis.axvline( x = upbound, color = 'b', label = '2.5% ->')
#axis.text(upbound, np.max(aehist[0]), '2.5% ->',**font)
#axis.axvline( x = lowbound, color = 'b', label = '<- 2.5%')
#axis.text(lowbound, np.max(aehist[0]), '<- 2.5%', horizontalalignment='right',**font)
#axis.set_title('avg y-axis = %1.3f' %avgeb + ' , std y-axis = %1.3f' %stdeb + ',  avg y-axis = %1.3f' %avgeb + ' , std y-axis = %1.3f' %stdeb ,**font)


#axis.set_xlabel('(sample intercept)-(population intercept)',**font)
#axis.set_ylabel('(sample slope)-(population slope)',**font)

