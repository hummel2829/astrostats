





import random
import matplotlib.pyplot as plt
from math import factorial
from math import exp

days = 1000
flarecount = 900
hoursinday = 24
successes = [*range(0,900)]

flaresim = flarecount*[1] + (days*hoursinday)*[0]

pflareinday = flarecount/(days*hoursinday)

flares = []

for i in range(0,days):
    flares = flares + [sum(random.choices(flaresim,k = (hoursinday)))]


print(sum(flares))


def number(n):
    return flares.count(n)/sum(flares)

result = map(number,successes)

flaresdist = list(filter(lambda num: num>0 , result))

def Pp(k):
    return((pflareinday*hoursinday)**k)*(exp(-pflareinday*hoursinday))/factorial(k)

result = map(Pp,successes[0:len(flaresdist)])
theoPp = list(result)


def PB(k):
    return((factorial(hoursinday)*(pflareinday**k)*((1-pflareinday)**(hoursinday-k))/(factorial(k)*factorial(hoursinday-k))))

result = map(PB,successes[0:len(flaresdist)])
theoPB = list(result)


lambdat = 1/(sum(flares)/len(flares))



def newdist(k):
    return(lambdat*k)

timedist = list(map(newdist,flaresim))

times = []

for i in range(0,days):
    times = times + [sum(random.choices(timedist,k = (hoursinday)))]






fig = plt.figure(1)
ax1 = fig.add_subplot(111)

ax1.bar(successes[0:len(flaresdist)],flaresdist, color = 'w', edgecolor = 'b' , width = 0.4, label='Sim')
ax1.scatter(successes[0:len(flaresdist)],theoPp, s=200, c='r', marker="3", label='Pp(k)')
ax1.scatter(successes[0:len(flaresdist)],theoPB, s=100, c='g', marker="+", label='PB(k)')

plt.legend(loc='upper right')



font = {'fontname' : 'Times New Roman' , 'size' : 20}
plt.title('Solar flares over 1000 days',**font)
plt.xlabel('number of flares per day',**font)
plt.ylabel('P',**font)





plt.show()


