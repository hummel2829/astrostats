import numpy as np
import random

N = 10000 # number of experiments
color = 1 # 1, 2, or 3

redballs   = 4*[1]
blueballs  = 5*[2]
greenballs = 6*[3]

balls = redballs + blueballs + greenballs

totalballs = N*[balls]

counta = 0
countb = 0
countc = 0
ballsdrawn = 3

'''
for i in range(0,N):
    balls = redballs + blueballs + greenballs
    print(balls)
    balls.remove(1)
    print(balls)

'''


for i in range(0,N):
    balls = redballs + blueballs + greenballs
    ball1 = random.choices(balls,k=1)
    balls.remove(ball1[0])
    #print(ball1 , balls)
    ball2 = random.choices(balls,k=1)
    balls.remove(ball2[0])
    #print(ball2 , balls)
    ball3 = random.choices(balls,k=1)
    balls.remove(ball3[0])
    #print(ball3 , balls)
    selections = [ball1 , ball2 , ball3]
    
    if selections[0] == selections[1] == [color] or \
       selections[1] == selections[2] == [color] or \
       selections[2] == selections[0] == [color]:
       counta = counta + 1
       
    if selections[0]==selections[1]==selections[2]:
       countb = countb + 1

    if selections[0] != selections[1] and \
       selections[1] != selections[2] and \
       selections[2] != selections[0]:
        countc = countc +1


print(counta/N) # 2/3 red
print(countb/N) # 3/3 same color
print(countc/N) # all diff colors
