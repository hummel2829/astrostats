





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






figure, axis = plt.subplots(2, 1,constrained_layout=True)

axis[0].bar(successes[0:len(flaresdist)],flaresdist, color = 'w', edgecolor = 'b' , width = 0.4, label='Sim')
axis[0].scatter(successes[0:len(flaresdist)],theoPp, s=200, c='r', marker="3", label='Pp(k)')
axis[0].scatter(successes[0:len(flaresdist)],theoPB, s=100, c='g', marker="+", label='PB(k)')

axis[0].legend(loc='upper right')



font = {'fontname' : 'Times New Roman' , 'size' : 20}
axis[0].set_title('Solar flares over 1000 days',**font)
axis[0].set_xlabel('number of flares per day',**font)
axis[0].set_ylabel('P',**font)

axis[1].hist(times)


font = {'fontname' : 'Times New Roman' , 'size' : 20}
axis[1].set_title('time between flares',**font)
axis[1].set_xlabel('days between flares',**font)
axis[1].set_ylabel('counts',**font)


plt.show()
figure.savefig('HW4_3.pdf',format = 'svg', dpi = 1000)



