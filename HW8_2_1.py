
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

c = np.zeros([N,len(z[0])])
for x in range(0,N):
    for k in range(0,len(z[0])):
        #print(k,'--')
        i = 0
        s = 0
        while i+k < len(z[0]):
            s = s + (z[x][i] - np.mean(z))*(z[x][i+k] - np.mean(z))
            #print(i, i+k)
            i = i + 1
        c[x][k] = s/len(z[0])
        
r = c/c[0]


p = np.zeros(5)
p[0] = 1
p[1] = a1/(1-a2)
p[2] = a2 + a1*p[1]
p[3] = a1*p[2] + a2*p[1]
p[4] = a1*p[2]

r1 = r[:,0]
rboot = np.random.choice(r1, size = N, replace = True)

# 99% CI -> z_0.005 -> z = +/- 2.33

ztest = (np.mean(rboot) - np.mean(p))/(np.std(rboot))
print(ztest)


# question 4

varr = np.zeros(len(p))
for k in range(0,len(p)):
    s = 0
    v = 0
    while v-k > 0:
        #s = s + p[v]**2 + p[v+k]*p[v-k] - 4*p[k]*p[v]*p[v-k] + 2*p[v]*p[k]*p[v]*p[k]
        v = v + 1
    varr[k] = s/len(p)



# 8_2_2

wa = np.zeros([N,2])
for i in range(0,N):
    wa[i][0] = z[i][1]*(1 - z[i][2]) / (1 - z[i][1]**2)
    wa[i][1] = (z[i][2] - z[i][1]**2) / (1 - z[i][1]**2)

# 99% CI -> z_0.005 -> z = +/- 2.33





figure1, axis = plt.subplots(1, 1,constrained_layout=True)
axis.hist(wa[:,0])
#axis.plot(r, color = "red", label='k')
#axis.plot(p, color = "blue", label='p')


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

