


import numpy as np
import matplotlib.pyplot as plt

    
    

'''--------------1.1.2b #1--------------'''

X = []

n = 100
draws = 10000

for i in range(0,draws):
    #print(i)
    C = np.random.randn(1,draws)
    Xbar = np.mean(C)
    X = np.append(X,Xbar)
    
plt.hist(X)

onetick= []


for i in range(0,draws):
    if X[i] <= 0.01 and X[i] >=-0.01:
        onetick=np.append(onetick,X[i])
        
print(len(onetick)/draws)

'''---------------1.1.2b #2-------------------'''


import numpy as np
import matplotlib.pyplot as plt


ratioplot = []
runsplot = []
drawsplot =[]
X = []
runs = 20
n = 100
draws = 10000
while runs>0:

    for i in range(0,int(draws)):
        C = np.random.randn(1,int(draws))
        Xbar = np.mean(C)
        X = np.append(X,Xbar)



    onetick= []


    for i in range(0,int(draws)):
        if X[i] <= 0.01 and X[i] >=-0.01:
            onetick=np.append(onetick,X[i])
    
    a = (len(onetick)/int(draws))
    
    ratioplot = np.append(ratioplot,a)
    runsplot = np.append(runsplot,runs)
    drawsplot = np.append(drawsplot,draws)
    runs = runs-1
    draws = draws -500
    
    
    
plt.scatter(drawsplot,ratioplot)

plt.title('ratio between [-0.1,0.1] response to sample size')
plt.xlabel('number of samples' )
plt.ylabel('ratio values')


'''---------------1.1.2b #3---------------------------'''


ER = []
ep=0.01
for i in range(0,int(draws)):
    ER=[]
    print(i)
    for a in range(0,int(draws)):
        if X[a]<=ep and X[a]>=-ep:
            ER=np.append(ER,X[a])
        if (len(ER)/draws)==0.99:
            #print('break1',a,i,ep)
            break
            
           
    ep=ep*1.1
    print(ep)
    #print(len(ER))
    if (len(ER)/draws)==0.99:
        print(i,len(ER)/draws,ep)
        break

print('epsilon range +/-',ep)
print('% of draws',len(ER)/draws)



'''---------------1.1.2b #4----------------------------'''

import numpy as np
import matplotlib.pyplot as plt



epsplot =[]
drawsplot = []
nplot = []
draws=10000
n = 100
runs=20
while runs>1:
    
    X = []


    for i in range(0,int(draws)):
        C = np.random.randn(1,int(draws))
        Xbar = np.mean(C)
        X = np.append(X,Xbar)
    

    onetick= []


    for i in range(0,int(draws)):
        if X[i] <= 0.01 and X[i] >=-0.01:
            onetick=np.append(onetick,X[i])
        
    epsplot = np.append(epsplot,len(onetick)/int(draws))
    drawsplot = np.append(drawsplot,draws)
    draws = draws-500
    runs = runs-1
    
    
    
    plt.scatter(drawsplot,epsplot)
    plt.title('epsilon response to sample size')
    plt.xlabel('number of samples' )
    plt.ylabel('epsilon values')
    
    









