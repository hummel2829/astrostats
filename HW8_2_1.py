
import matplotlib.pyplot as plt
import numpy as np

N = 5

z = np.array([1,1])

a1 = 0.7
a2 = -0.2
start = len(z)
for i in range(start,N+1):
    z = np.append(z,(a1*z[i-1] + a2*z[i-2] + np.random.normal(0,1,1)))
    
c = np.zeros(N)
s = 0
for k in range(0,N):
    for n in range(0,N-k):
       s = s + (z[n] - np.mean(z))*(z[n+k] - np.mean(z))
       c[k] = (1/(N-k))*s
       
       
r = c/c[0]








figure1, axis = plt.subplots(1, 1,constrained_layout=True)
#axis.scatter(r,, s=20, c='r', marker="o", label='exp(k)')
axis.plot(r)


axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

axis.axvline( x = upbound, color = 'b', label = '2.5% ->')
axis.text(upbound, np.max(aehist[0]), '2.5% ->',**font)
axis.axvline( x = lowbound, color = 'b', label = '<- 2.5%')
axis.text(lowbound, np.max(aehist[0]), '<- 2.5%', horizontalalignment='right',**font)
#axis.set_title('avg y-axis = %1.3f' %avgeb + ' , std y-axis = %1.3f' %stdeb + ',  avg y-axis = %1.3f' %avgeb + ' , std y-axis = %1.3f' %stdeb ,**font)


#axis.set_xlabel('(sample intercept)-(population intercept)',**font)
#axis.set_ylabel('(sample slope)-(population slope)',**font)



plt.grid()
plt.show()

