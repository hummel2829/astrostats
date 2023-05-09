
import matplotlib.pyplot as plt
import numpy as np

filename = r'D:\Python\picoquant\antibunch\NV-Center_for_Antibunching_1.out'
NVdata = np.loadtxt(filename, delimiter = ',', usecols = (0,1,2), skiprows = 1)
NVdata = NVdata.astype(np.int64)


ch1count = np.count_nonzero(NVdata[:,0] == 1)


ch1index = np.nonzero(NVdata[:,0] == 1)

NVch1TimeTag = NVdata[ch1index,1]
NVch1TrueTime = NVdata[ch1index,2]

NVch1TrueTimet1stdiff = np.diff(NVch1TrueTime)
x1stdiff = np.arange(0,NVch1TrueTimet1stdiff.shape[1])

photonpersecond = len(NVch1TrueTime[0]) / NVch1TrueTime[0][-1] # number photon / total measurement time (picoseconds)

bins=np.arange(0,100,1)

intervaldiff = 1e7

ch1TrueTimehist , bb = np.histogram(NVch1TrueTimet1stdiff/intervaldiff,bins=bins)

avgexp_unitless = (np.sum(NVch1TrueTimet1stdiff/intervaldiff)) / len(NVch1TrueTimet1stdiff[0])

ch1TrueTimehist = ch1TrueTimehist/np.sum(ch1TrueTimehist)

ch1TrueTimehist = ch1TrueTimehist[1:]

logch1TrueTimehist = np.log(ch1TrueTimehist)
idx = logch1TrueTimehist != -np.inf
logch1TrueTimehist = logch1TrueTimehist[idx]
binsc = bins[1:-1]+0.5
binscidx = binsc[idx]
#binscidx = binscidx[1:]



x = binscidx #normalized deltat/interval difference

y = logch1TrueTimehist #normalized for probability instead of counts

mfit,bfit = np.polyfit(x , y ,1)

############## paper MLE

x = np.sort(NVch1TrueTime[0])/intervaldiff

lamvalue = np.array([0.1 ,-mfit]) # -mfit bc lambda negative in equation
i = len(lamvalue)
while (np.abs(lamvalue[i-1]-lamvalue[i-2])/lamvalue[i-2]) > 0.0001:

    lamiteration =( np.exp(-lamvalue[i-1]*x[0]) - np.exp(-lamvalue[i-1]*x[-1]) ) \
    / ( (avgexp_unitless - x[0]) * np.exp(-lamvalue[i-1]*x[0]) \
     - (avgexp_unitless - x[-1]) * np.exp(-lamvalue[i-1]*x[-1]) )
        
    lamvalue = np.append(lamvalue,lamiteration)
    i = i + 1
    
#################################


###################### john edit to wolf y -> lny
x = binscidx #normalized deltat/interval difference

y = logch1TrueTimehist #normalized for probability instead of counts

mj = (np.sum(np.log(y)) * np.sum(x * np.log(y) * np.log(y)) - np.sum(x*np.log(y)) * np.sum(np.log(y)*np.log(y))) \
    / ( np.sum(np.log(y)) * np.sum(x * x * np.log(y)) - np.sum(x * np.log(y))**2 )


bj = ( np.sum(x * x * np.log(y)) * np.sum(np.log(y)*np.log(y)) - np.sum(x * np.log(y)) * np.sum(x * np.log(y) * np.log(y)) ) \
    / ( np.sum(np.log(y)) * np.sum(x * x * np.log(y)) - np.sum(x * np.log(y))**2 )


##############################################

#################### poisson sim
lambda1 = (1/avgexp_unitless) # 1/ (time per photon)

lambda2 = -mfit # slope naive line fit = -m 

lambda3 = np.exp(bfit)  # = intercept naive line fit = e^b 

lambda4 = lamvalue[-1] # iteration method from paper considering weighted regression

lambda5 = mj

lambda6 = bj

lambdaall = np.array([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6])


measurements = 10000

#nanobysec = (1e9) #nanoseconds in sec, convert photon/pico to photon/milisec

pdata = np.random.poisson(lambdaall,(measurements,len(lambdaall)))

pdata1 = pdata[:,0]
pd1 = np.reshape(pdata1,(100,100))

pdata2 = pdata[:,1]
pd2 = np.reshape(pdata2,(100,100))

pdata3 = pdata[:,2]
pd3 = np.reshape(pdata3,(100,100))

pdata4 = pdata[:,3]
pd4 = np.reshape(pdata4,(100,100))


w1 = np.ones(len(pdata1))/len(pdata1)
pdata1prob, pdata1binedge = np.histogram(pdata1,bins=np.max(pd1)+1, weights = w1)

w2 = np.ones(len(pdata2))/len(pdata2)
pdata2prob, pdata2binedge = np.histogram(pdata2,bins=np.max(pd2)+1, weights = w2)

w3 = np.ones(len(pdata3))/len(pdata3)
pdata3prob, pdata3binedge = np.histogram(pdata3,bins=np.max(pd3)+1, weights = w3)

w4 = np.ones(len(pdata4))/len(pdata4)
pdata4prob, pdata4binedge = np.histogram(pdata4,bins=np.max(pd4)+1, weights = w4)



############# exp sim

t = binsc

exp1 = lambda1*np.exp(-lambda1*t)

exp2 = lambda2*np.exp(-lambda2*t)

exp3 = lambda3*np.exp(-lambda3*t)

exp4 = lambda4*np.exp(-lambda4*t)


#################################



figure1, axis = plt.subplots(1, 1,constrained_layout=True)

axis.scatter(binscidx , logch1TrueTimehist, s=50, c='r', marker="o", label='interval count')


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font)
axis.set_ylabel('log(pmf)',**font)


figure2, axis = plt.subplots(1, 1,constrained_layout=True)

axis.plot(binsc , np.log(exp1) - np.log(ch1TrueTimehist),  c='r', marker="o", label='1/mean fit,  lambda1 = %1.5f' %lambda1)
axis.plot(binsc , np.log(exp2) - np.log(ch1TrueTimehist),  c='b', marker="o", label='slope of lin reg, lambda2 = %1.5f' %lambda2)
axis.plot(binsc , np.log(exp3) - np.log(ch1TrueTimehist),  c='g', marker="*", label='intercept of lin reg, lambda3 = %1.5f' %lambda3)
axis.plot(binsc , np.log(exp4) - np.log(ch1TrueTimehist),  c='purple', marker="X", label='iteration of lambda, lambda4 = %1.5f' %lambda4)
#axis.plot(binsc , np.log(ch1TrueTimehist), label='interval count')
axis.legend(loc='upper left',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('Simulated exp distributions error' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font)
axis.set_ylabel('log(exp fit) - log(data)',**font)



figure3, axis = plt.subplots(1, 1,constrained_layout=True)
x1 = pdata1binedge[0:-1]+(pdata1binedge[1] - pdata1binedge[0])/2
x2 = pdata2binedge[0:-1]+(pdata2binedge[1] - pdata2binedge[0])/2
x3 = pdata3binedge[0:-1]+(pdata3binedge[1] - pdata3binedge[0])/2
x4 = pdata4binedge[0:-1]+(pdata4binedge[1] - pdata4binedge[0])/2

x1 = np.arange(4)
x2 = x3 = x4 = x1

size = 200

axis.scatter( x1, pdata1prob, s = size, color = 'blue', label = 'lambda1 = %1.5f  ' %lambda1 + ',  mean counts = %1.5f' %np.mean(pdata1))
axis.scatter( x2, pdata2prob, s = size, color = 'red', label = 'lambda2 = %1.5f  ' %lambda2 + ',  mean counts = %1.5f' %np.mean(pdata2))
axis.scatter( x3, pdata3prob, s = size, color = 'orange', label = 'lambda3 = %1.5f  ' %lambda3 + ',  mean counts = %1.5f' %np.mean(pdata3))
axis.scatter( x4, pdata4prob, s = size, color = 'purple', label = 'lambda4 = %1.5f  ' %lambda4 + ',  mean counts = %1.5f' %np.mean(pdata4))


axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('poisson sim from rate parameter estimations',**font)

axis.set_xlabel('photons detected',**font)
axis.set_ylabel('pmf',**font)


figure4, axis = plt.subplots(1, 1,constrained_layout=True)

axis.plot(binsc , np.log(exp1),  c='r', marker="o", label='1/mean fit,  lambda1 = %1.5f' %lambda1)
axis.plot(binsc , np.log(exp2),  c='b', marker="o", label='slope of lin reg, lambda2 = %1.5f' %lambda2)
axis.plot(binsc , np.log(exp3),  c='g', marker="*", label='intercept of lin reg, lambda3 = %1.5f' %lambda3)
axis.plot(binsc , np.log(exp4),  c='purple', marker="X", label='iteration of lambda, lambda4 = %1.5f' %lambda4)
axis.plot(binsc , np.log(ch1TrueTimehist), label='interval count')
axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('Simulated exp distributions error' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font)
axis.set_ylabel('log(exp fit) - log(data)',**font)




plt.grid()
plt.show()



