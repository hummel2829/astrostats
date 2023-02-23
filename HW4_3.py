





import random
import matplotlib.pyplot as plt
from math import factorial
from math import exp

experiments = 1000   
days = 1000
flarecount = 900
hoursinday = 24
successes = [*range(0,900)]

flaresim = flarecount*[1] + (days*hoursinday)*[0]

pflareinday = flarecount/(days*hoursinday)

flares = []

for i in range(0,experiments):
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



fig = plt.figure()
ax1 = fig.add_subplot(111)

#plt.bar(successes[0:len(flaresdist)],flaresdist,width = 0.05)


ax1.bar(successes[0:len(flaresdist)],flaresdist,width = 0.05, label='Sim')
ax1.scatter(successes[0:len(flaresdist)],theoPp, s=200, c='r', marker="3", label='Pp(k)')
ax1.scatter(successes[0:len(flaresdist)],theoPB, s=20, c='g', marker="o", label='PB(k)')

plt.legend(loc='upper right')



font = {'fontname' : 'Times New Roman' , 'size' : 20}
plt.title('title',**font)
plt.xlabel('number of flares',**font)
plt.ylabel('P',**font)
plt.show()


