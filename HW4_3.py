





import random
import matplotlib.pyplot as plt
import numpy as np

experiments = 1000   
days = 1000
flares = 900
hoursinday = 24
successes = [*range(0,900)]
successes = [*range(0,9)]

flaresim = 900*[1] + (days*hoursinday)*[0]

pflareinday = flares/(days*hoursinday)

flares = []

for i in range(0,experiments):
    flares = flares + [sum(random.choices(flaresim,k = (hoursinday)))]

print(sum(flares))
counts , bins , bars = plt.hist(flares, density = True)


font = {'fontname' : 'Times New Roman' , 'size' : 20}
plt.title('title',**font)
plt.show()
print(sum(counts))


