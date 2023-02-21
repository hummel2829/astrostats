





import random
import matplotlib
import math


   
days = 1000
flares = 900
hoursinday = 24
successes = [*range(0,900)]
successes = [*range(0,9)]

flaresim = 900*[1] + (days*hoursinday)*[0]

pflareinday = flares/(days*hoursinday)

flaredata = random.choices(flaresim,k = (days*hoursinday))

countflare = 0

for i in range(0,len(flaredata)):
    if flaredata[i] == 1:
        countflare = countflare + 1



print(countflare)