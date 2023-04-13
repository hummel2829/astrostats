
import datetime
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from math import exp
from math import ceil

debug = False


filename = 'xray.txt'
xraydata = np.loadtxt(filename)
xraydata = xraydata.astype(int)

xraydata = xraydata[0:10,:]
#import numpy as np
#xraydata = np.arange(24*60*3,dtype=np.int_)
#xraydata = [i for i in range(0,24*60*3)]

rows =[*range(0,xraydata.shape[0])]
if debug:
    print(rows)
dates = [datetime.datetime(*x) for x in xraydata]
if debug:
    print(dates)
success = [*range(0,len(dates))]

#minutes from first flare
minfrom1stflare = [(((dates[x] - dates[0]).days)*24*60) + ((dates[x] - dates[0]).seconds)//60 for x in rows]

allminutes = np.zeros(minfrom1stflare[-1]+1)

np.put(allminutes, minfrom1stflare, 1)

flares = allminutes.astype(int)

mininday = int(24*60)

flaresdayarray = np.copy(flares)
flaresdayarray.resize(ceil(minfrom1stflare[-1]/mininday),mininday)

#flaressuminday = np.sum(flaresdayarray,axis=1)

flaressuminday = np.sum(np.random.binomial(1,0.7,(flaresdayarray.shape[0]*flaresdayarray.shape[1])).reshape((flaresdayarray.shape[0],flaresdayarray.shape[1])),axis=1)

w = flaressuminday/len(flaressuminday)
flaredayhist = np.histogram(flaressuminday,bins=len(flaressuminday))


#Binomial dist
N = int(np.max(flaredayhist[1])+1)
#p = np.max(flaredayhist[0])
p = np.sum(flares)/len(allminutes)
def PB(k):
    return((factorial(N)/(factorial(k)*factorial(N-k)))*(p**k)*(1-p)**(N-k))
theoPB = list(map(PB,success[0:N]))
#theoPB = list(result)


# poisson
def Pp(k):
    return(((p*N)**k)*(exp(-p*N))/factorial(k))
theoPp = list(map(Pp,success[0:N]))

#exponential

y = np.sum(flares)/(len(allminutes)/(mininday))
def Pe(k):
    return(y*(exp(-y*k)))
theoPe = list(map(Pe,success[0:N]))





figure, axis = plt.subplots(1, 1,constrained_layout=True)


hist = axis.hist(flaressuminday,bins = int(np.max(flaredayhist[1])), weights = w, color = 'w', edgecolor = 'b' , label='Sim')
#axis.plot(success[0:N],theoPB, c='r', marker="o", label='PB(k)')
#axis.plot(success[0:N],theoPp, c='b', marker="o", label='Pp(k)')
#axis.plot(success[0:N],theoPe[0:N], c='orange', marker="o", label='Pe(k)')


font = {'fontname' : 'Times New Roman' , 'size' : 20}
axis.legend(loc='upper right',fontsize = 20)
#axis.set_title('avg = %1.3f' %avg + ' , var = %1.3f' %stdN ,**font)

axis.set_title('exp rate parameter = 7.16',**font)
axis.set_xlabel('flares per day',**font)
axis.set_ylabel('P',**font)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

plt.show()




'''

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

'''


