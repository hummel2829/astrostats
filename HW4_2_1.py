import matplotlib.pyplot as plt
import numpy as np


N = 10000
n = 10

stdset = []

for i in range(0,N):
    stdset.append(np.var(np.random.normal(loc = 0, scale = 1.0, size = n)))

avg = sum(stdset)/len(stdset)

def f(x):
    return (x - avg)**2
stdN = sum(list(map(f,stdset)))/N



fig = plt.figure(1)
ax1 = fig.add_subplot(111)

ax1.hist(stdset)


font = {'fontname' : 'Times New Roman' , 'size' : 20}
plt.title('avg = %1.3f' %avg + ' , var = %1.3f' %stdN ,**font)
plt.xlabel('var values',**font)
plt.ylabel('number of var',**font)

plt.show()

fig.savefig('HW4_2_2.pdf',format = 'svg', dpi = 1000)




