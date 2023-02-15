import numpy as np
import random

N = 10000 # number of experiments
color = 1 # 1, 2, or 3

redballs   = 4*[1]
blueballs  = 5*[2]
greenballs = 6*[3]

balls = redballs + blueballs + greenballs

totalballs = N*4*[1] + N*5*[2] + N*6*[3]

counta = 0
countb = 0
countc = 0
ballsdrawn = 3

for i in range(0,N):
    selections = random.choices(balls,k=ballsdrawn)
    if selections[0]==selections[1]==color or \
       selections[1]==selections[2]==color or \
       selections[2]==selections[0]==color:
       counta = counta + 1
       balls.remove(selections[0])
       balls.remove(selections[1])
       balls.remove(selections[2])
    if selections[0]==selections[1]==selections[2]:
       countb = countb + 1
       balls.remove(selections[0])
       balls.remove(selections[1])
       balls.remove(selections[2])
    if selections[0] != selections[1] and \
       selections[1] != selections[2] and \
       selections[2] != selections[0]:
        countc = countc +1
        balls.remove(selections[0])
        balls.remove(selections[1])
        balls.remove(selections[2])
        #print(selections)

print(counta/N) # 0.1751 - not matching analytic result
print(countb/N) # 0.1248
print(countc/N) # 0.2119
