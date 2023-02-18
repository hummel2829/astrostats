import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import random
import math


coin =[-1,1]
exp =10000
steps = 100
oddsteps = [*range(1,steps,2)]
evensteps = [*range(0,steps,2)]
stepstaken = 100


finalpos = []

for i in range(0,exp):
    trial = sum(random.choices(coin,k=steps))
    finalpos.append(trial)
    
 
'''   
 
evenfinalpos = []
theofinalprob = []
for i in range(0,steps//2):
    oddstepsfromorigin = oddsteps
    evenstepsfromorigin = evensteps
    for a in range(0,i):
        even = (0.5**i)*math.factorial(evensteps[i])//(math.factorial((evensteps[a]-evenstepsfromorigin[a])//2) + math.factorial((evensteps[a]-evenstepsfromorigin[a])//2))
        theofinalprob.append(even)
    
    
'''    
    

counts,bins,bars = plt.hist(finalpos,bins= np.linspace(-40, 40, num=100),range=(-10,10),align='mid', density = True,stacked = True)
print(max(counts))

counts,bins,bars = plt.hist(finalpos,bins= np.linspace(-40, 40, num=100),range=(-10,10),align='mid')
print(max(counts)/10000)

plt.title('final positions from 10000 random walk of 100 steps')
plt.xlabel('final positions')
plt.ylabel('count of final positions')




