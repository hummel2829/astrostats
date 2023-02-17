import numpy as np
import matplotlib.pyplot as plt
import random
import math


coin =[-1,1]
exp =10000
steps = 100
stepstaken = 100


finalpos = []

for i in range(0,exp):
    trial = sum(random.choices(coin,k=steps))
    finalpos.append(trial)
    
    
for i in range(0,steps):
    stepsfromorigin = steps
    for a in range(0,stepsfromorigin):
        if float.is_integer((steps-stepsfromorigin)/2)  or \
            float.is_integer((steps+stepsfromorigin)/2):
                math.factorial(steps)/(math.factorial((steps-stepsfromorigin)/2) + math.factorial((steps-stepsfromorigin)/2))
        
    


y,x,_ =plt.hist(finalpos,bins= np.linspace(-40, 40, num=100),range=(-10,10),align='mid')
#plt.xticks(list(range(-10,10)))
plt.title('final positions from 10000 random walk of 100 steps')
plt.xlabel('final positions')
plt.ylabel('count of final positions')

