import numpy as np
import matplotlib.pyplot as plt

'''--------------1.1.2b #1--------------'''

# Put option vars on top.
n = 100
draws = 10000

X = []
for i in range(0, draws):
    C = np.random.randn(1, draws)
    Xbar = np.mean(C)
    X = np.append(X, Xbar)

plt.hist(X)

onetick = []
for i in range(0, draws):
    if X[i] <= 0.01 and X[i] >= -0.01:
        onetick=np.append(onetick, X[i])

print(len(onetick)/draws)

'''---------------1.1.2b #2-------------------'''

# No need to reimport, but sometimes convenient if you expect people
# to copy/past only this part.
import numpy as np
import matplotlib.pyplot as plt

runs = 20
n = 100
draws = 10000

ratioplot = []
runsplot = []
drawsplot = []
X = []
while runs > 0:
    for i in range(0,int(draws)):
        C = np.random.randn(1,int(draws))
        Xbar = np.mean(C)
        X = np.append(X,Xbar)
    onetick= []


    for i in range(0,int(draws)):
        if X[i] <= 0.01 and X[i] >= -0.01:
            onetick=np.append(onetick,X[i])

    a = (len(onetick)/int(draws))

    ratioplot = np.append(ratioplot,a)
    runsplot = np.append(runsplot,runs)
    drawsplot = np.append(drawsplot,draws)
    runs = runs - 1
    draws = draws - 500 # Avoid modifying option vars.

plt.scatter(drawsplot,ratioplot)

plt.title('ratio between [-0.1,0.1] response to sample size')
plt.xlabel('number of samples' )
plt.ylabel('ratio values')


'''---------------1.1.2b #3---------------------------'''

draws = 10000
ep = 0.01 # epsilon
# (when you define a variable, sometimes it is helpful to reader if you
# state what the abbreviation means.)

ER = []
for i in range(0,int(draws)):
    ER=[]
    print(i)
    for a in range(0,int(draws)):
        if X[a]<=ep and X[a]>=-ep:
            ER=np.append(ER,X[a])
        if (len(ER)/draws)==0.99:
            #print('break1',a,i,ep)
            break
    ep = ep*1.1
    print(ep)
    #print(len(ER))
    if (len(ER)/draws) == 0.99:
        print(i,len(ER)/draws,ep)
        break

print('epsilon range +/-', ep)
# Be more explicit with print statements. 
print('% of draws in epsilon range', len(ER)/draws)


'''---------------1.1.2b #4----------------------------'''

import numpy as np
import matplotlib.pyplot as plt

draws = 10000
n = 100
runs = 20

epsplot = []
drawsplot = []
nplot = []
while runs > 1:

    X = []

    for i in range(0,int(draws)):
        C = np.random.randn(1,int(draws))
        Xbar = np.mean(C)
        X = np.append(X,Xbar)

    onetick = []

    for i in range(0,int(draws)):
        if X[i] <= 0.01 and X[i] >=-0.01:
            onetick = np.append(onetick,X[i])

    epsplot = np.append(epsplot,len(onetick)/int(draws))
    drawsplot = np.append(drawsplot,draws)
    draws = draws - 500
    runs = runs-1

    plt.scatter(drawsplot,epsplot)
    # Title essentially states y and x labels, so redundant. Use space
    # to tell reader something more (or exclude).
    plt.title('$\epsilon$ dependence on sample size')
    plt.xlabel('number of samples' )
    plt.ylabel('$\epsilon$ values')
