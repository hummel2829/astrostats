import random

N = 10000 # number of experiments
color = 1 # 1, 2, or 3

redballs   = 4*[1]
blueballs  = 5*[2]
greenballs = 6*[3]

counta = 0
countb = 0
countc = 0
ballsdrawn = 3


for i in range(0,N):                            #loop removes balls
    balls = redballs + blueballs + greenballs   #selected then checks
    ball1 = random.choices(balls,k=1)           #conditions for tallying
    balls.remove(ball1[0])

    ball2 = random.choices(balls,k=1)
    balls.remove(ball2[0])

    ball3 = random.choices(balls,k=1)
    balls.remove(ball3[0])

    selections = [ball1 , ball2 , ball3]
    
    if selections[0] == selections[1] == [color] or \
       selections[1] == selections[2] == [color] or \
       selections[2] == selections[0] == [color]:     #check for
       counta = counta + 1                            #2/3 red
       
    if selections[0]==selections[1]==selections[2]: #checks for
       countb = countb + 1                          #all same color

    if selections[0] != selections[1] and \
       selections[1] != selections[2] and \
       selections[2] != selections[0]:              #checks for
        countc = countc +1                          #all diff color


print(counta/N) # 2/3 red          0.1501
print(countb/N) # 3/3 same color   0.0763
print(countc/N) # all diff colors  0.2627