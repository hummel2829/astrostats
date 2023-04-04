
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
'''
emitters = 10
detect = np.array([0,1])

photondist = np.random.choice(detect , size = emitters, replace = True , p=[0.5,0.5])

'''
measurements = 10000

#poisson sim

photonrate = 3e8 #photon/second
timeinterval = 1/(3e8) #seconds
lam = photonrate*timeinterval  
pdata = np.random.poisson(lam,measurements)

pd = np.reshape(pdata,(100,100))

pdbootmean = np.average(np.random.choice(pdata,[10000,20]),axis=1)

#Hnull is pdbootmean == stdpdata
#Halt is pdbootmean not= stdpdata
#alpha = 99.5, Z table value 2.58, significance level 0.01

zpd = (np.mean(pdbootmean) - np.std(pdata))/(np.std(pdbootmean)/(10000**0.5))

# -2.58 < zpd=1.34 < 2.59 so cannot reject Hnull

pdmeanhist = np.histogram(pdbootmean, bins = 100)

pstd = np.std(pdata)

pdatahist = np.histogram(pdata,bins=np.max(pdata)+1)

#sub poisson sim

photonrate = 3e8 #photon/second
timeinterval = 1/(1e6) #seconds
lam = photonrate*timeinterval  
subpdata = np.random.poisson(lam,measurements)

subpd = np.reshape(subpdata,(100,100))


subpdbootmean = np.average(np.random.choice(subpdata,[10000,20]),axis=1)

#Hnull is subpdbootmean == stdsubpdata
#Halt is subpdbootmean not= stdsubpdata
#alpha = 99.5, Z table value 2.58, significance level 0.01

zsubpd = (np.mean(subpdbootmean) - np.std(subpdata))/(np.std(subpdbootmean)/(10000**0.5))

# -2.58 < zsubpd = 7244 so can reject Hnull

subpdatahist = np.histogram(subpdata,bins=np.max(subpdata)+1)





#super poisson sim

photonrate = 3e8 #photon/second
timeinterval = 1/(1e6) #seconds
lam = photonrate*timeinterval  
superpdata = np.random.poisson(lam,measurements)

superpd = np.reshape(superpdata,(100,100))


superpdbootmean = np.average(np.random.choice(superpdata,[10000,20]),axis=1)

#Hnull is subpdbootmean == stdsubpdata
#Halt is subpdbootmean not= stdsubpdata
#alpha = 99.5, Z table value 2.58, significance level 0.01

zsuperpd = (np.mean(superpdbootmean) - np.std(superpdata))/(np.std(superpdbootmean)/(10000**0.5))

# -2.58 < zsuperpd = 7369 so can reject Hnull





superpdatahist = np.histogram(superpdata,bins=np.max(superpdata)+1)





figure1, axis = plt.subplots(1, 1,constrained_layout=True)
#hdata, bins, bars = axis.hist(pdata,bins=np.max(pdata), align = 'mid')

#x = 0.5*(pdatahist[1][1:] + pdatahist[1][:-1])
x = np.arange(0,np.max(pdata)+1)
#axis.plot(x , pdatahist[0] , color="red")
axis.scatter(x,pdatahist[0], s=100, c='r', marker="o", label='poisson light')
axis.bar(x,pdatahist[0])

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





figure2, axis = plt.subplots(1, 1,constrained_layout=True)

#x = np.arange(0,np.max(pdata)+1)
#axis.plot(pdrowmean , color="red")
axis.hist(pdmean,bins=100)
#axis.scatter(x,pdrowmean, s=100, c='r', marker="o", label='poisson light')


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









plt.grid()
plt.show()







