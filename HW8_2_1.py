
import matplotlib.pyplot as plt
import numpy as np

N = 1000


a1 = 0.7
a2 = -0.2

z = np.zeros([N])
z[0] = np.random.normal(0,1,1)
z[1] = a1*z[0] + np.random.normal(0,1,1)

for i in range(2,N):
    z[i] = a1*z[i-1] + a2*z[i-2] + np.random.normal(0,1,1) 


    
c = np.zeros(N)
for k in range(0,N):
    s = 0
    t = 0
    while t+k < N:
        s = s + (z[t] - np.mean(z))*(z[t+k] - np.mean(z))
        t = t + 1
    c[k] = s/(N-1)
        
r = c/c[0]



p = np.zeros(N)


p[0] = 1
p[1] = a1/(1-a2)
p[2] = a2 + a1*p[1]

for m in range(3,N):
    p[m] = a1*p[m-1] + a2*p[m-2]
        


# 99% CI -> z_0.005 -> z = +/- 2.33


c1 = np.zeros(N)
t = 0
while t+1 < N:
    c1[t] = (1/(t+1))*(z[t] - np.mean(z))*(z[t+1] - np.mean(z))
    t = t + 1
    
        
r1 = np.cumsum(c1)/c[0]

r1boot = np.random.normal(np.mean(r1) , np.std(r1) , size = 1000)

ztestr1 = (np.mean(r1boot) - np.mean(r1))/(np.std(r1boot)/1000**0.5)
print(ztestr1)


# question 4

varr = np.zeros(len(p))
for k in range(0,len(p)):
    s = 0
    v = 0
    while v-k > 0:
        s = s + p[v]**2 + p[v+k]*p[v-k] - 4*p[k]*p[v]*p[v-k] + 2*p[v]*p[k]*p[v]*p[k]
        v = v + 1
    varr[k] = s/len(p)



# 8_2_2

c2 = np.zeros(N)
t = 0
while t+2 < N:
    c2[t] = (1/(t+1))*(z[t] - np.mean(z))*(z[t+2] - np.mean(z))
    t = t + 1
    
        
r2 = np.cumsum(c2)/c[0]

r2boot = np.random.normal(np.mean(r2) , np.std(r2) , size = 1000)


wa1 = r1*(1-r2)/(1-r1**2)
wa2 = (r2 - r1**2)/(1-r1**2)

wa1boot = np.random.normal(np.mean(wa1) , np.std(wa1) , size = 1000)
wa2boot = np.random.normal(np.mean(wa2) , np.std(wa2) , size = 1000)

# 99% CI -> z_0.005 -> z = +/- 2.33

ztestwa1 = (np.mean(wa1boot) - np.mean(a1))/(np.std(wa1boot)/1000**0.5)
print(ztestwa1)

ztestwa2 = (np.mean(wa2boot) - np.mean(a2))/(np.std(wa2boot)/1000**0.5)
print(ztestwa2)


# 99% CI -> z_0.005 -> z = +/- 2.33


#8_2_3

vareps = (1 - a1*p[1] + a2*p[2])*np.std(z)





figure1, axis = plt.subplots(1, 1,constrained_layout=True)
axis.hist(rboot)
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

