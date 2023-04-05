
import matplotlib.pyplot as plt
import numpy as np

N = 1000


a1 = 0.7
a2 = -0.2

z = np.zeros([N,5])

for i in range(0,N):
    z[i][0] = np.random.normal(0,1,1)
    z[i][1] = a1*z[i][0] + np.random.normal(0,1,1)
    z[i][2] = a1*z[i][1] + a2*z[i][0] + np.random.normal(0,1,1)
    z[i][3] = a1*z[i][2] + a2*z[i][1] + np.random.normal(0,1,1)
    z[i][4] = a2*z[i][2] + np.random.normal(0,1,1)

c = np.zeros(len(z[0]))

for k in range(0,len(z[0])):
    print(k,'--')
    i = 0
    s = 0
    while i+k < len(z[0]):
        s = s + (z[0][i] - np.mean(z))*(z[0][i+k] - np.mean(z))
        print(i, i+k)
        i = i + 1
    c[k] = s
        
r = c/c[0]

'''
z = np.array([0.5,0.5])
start = len(z)
for i in range(start,N+1):
    
    z = np.append(z,(a1*z[i-1] + a2*z[i-2] + np.random.normal(0,1,1)))
    
c = np.zeros(N)
s = 0
for k in range(0,N):
    for n in range(0,N-k):
       s = s + (z[n] - np.mean(z))*(z[n+k] - np.mean(z))
       c[k] = (1/(N-k))*s
'''       
       


#HW8_2_1 #2
#phi = np.zeros(r.shape[0])
#phi[0] = r[0]*(1 - r[1]) / (1 - r[1]**2)
#phi[1] = (r[1] - r[0]**2) / (1 - r[0]**2)

p1 = a1/(1-a2)
p2 = a2 + a1**2/(1-a2)


p = np.zeros(5)
p[0] = 1
p[1] = a1/(1-a2)
p[2] = a2 + a1*p[1]
p[3] = a1*p[2] + a2*p[1]
p[4] = a1*p[2]


pboot = np.random.choice(p, size = N, replace = True)





figure1, axis = plt.subplots(1, 1,constrained_layout=True)
#axis.scatter(r,, s=20, c='r', marker="o", label='exp(k)')
axis.plot(r, color = "red", label='k')
axis.plot(p, color = "blue", label='p')


axis.legend(loc='upper right')

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)


#axis.axvline( x = upbound, color = 'b', label = '2.5% ->')
#axis.text(upbound, np.max(aehist[0]), '2.5% ->',**font)
#axis.axvline( x = lowbound, color = 'b', label = '<- 2.5%')
#axis.text(lowbound, np.max(aehist[0]), '<- 2.5%', horizontalalignment='right',**font)
#axis.set_title('avg y-axis = %1.3f' %avgeb + ' , std y-axis = %1.3f' %stdeb + ',  avg y-axis = %1.3f' %avgeb + ' , std y-axis = %1.3f' %stdeb ,**font)


#axis.set_xlabel('(sample intercept)-(population intercept)',**font)
#axis.set_ylabel('(sample slope)-(population slope)',**font)



plt.grid()
plt.show()

