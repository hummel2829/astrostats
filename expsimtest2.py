
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon


#lambdatest = 0.1133
lambdatest = 1

expsimraw =np.sort( np.random.exponential(lambdatest , 10000))
A = np.max(expsimraw)
B = np.min(expsimraw)
expsimdata = expsimraw
#expsimdata = expsimdata[::-1] #/np.max(expsimdata)


t = np.arange(0,5,0.1)

expsci = expon.pdf(t)

expsim = lambdatest*np.exp(-lambdatest*t)

expsimchoice = np.random.choice(expsim, 10000)
plt.figure(1)

nphist, npbin, ign = plt.hist(expsimdata, color = 'red')

xnp = npbin[0:-1] + np.diff(npbin)[0]

jhist, jbin, ign = plt.hist(expsimchoice, color = 'blue')

xj = jbin[0:-1] + np.diff(jbin)[0]


plt.figure(2)

plt.plot(expsim, 'r', linewidth = 20)
plt.plot(expsci, 'g', linewidth = 5)

#plt.plot(expsimdata, 'r')

plt.figure(3)
plt.hist(expsimchoice, color = 'blue')
plt.hist(expsimdata, color = 'red', alpha = 0.5)


plt.figure(4)
plt.bar(xnp, nphist/np.sum(nphist), color = 'red')
#plt.bar(xj, jhist/np.sum(jhist), color = 'blue', alpha = 0.5)


plt.figure(5)
plt.plot(expsci, 'g', linewidth = 10)




plt.grid
plt.show


