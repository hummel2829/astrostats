import numpy as np

N = 10000 # number of experiments
color = 1 # 1, 2, or 3

redballs   = np.array([1,1,1,1])
blueballs  = np.array([2,2,2,2,2])
greenballs = np.array([3,3,3,3,3,3])

balls = np.concatenate([redballs,blueballs,greenballs])

counta = 0
countb = 0
countc = 0
selections = np.empty([N,3])
for i in range(0,N):
    selections = np.random.choice(balls,3)
    if selections[0]==selections[1]==color or \
       selections[1]==selections[2]==color or \
       selections[2]==selections[0]==color:
       counta = counta + 1
    if selections[0]==selections[1]==selections[2]:
       countb = countb + 1
    if selections[0] != selections[1] and \
       selections[1] != selections[2] and \
       selections[2] != selections[0]:
        countc = countc +1
        #print(selections)

print(counta/N) # 0.1751 - not matching analytic result
print(countb/N) # 0.1248
print(countc/N) # 0.2119

