
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

ch1TrueTimehist , bb = np.histogram(NVch1TrueTimet1stdiff/intervaldiff,bins=np.arange(0,100,1))

avgexp_unitless = (np.sum(NVch1TrueTimet1stdiff/intervaldiff)) / len(NVch1TrueTimet1stdiff[0])

ch1TrueTimehist = ch1TrueTimehist/np.sum(ch1TrueTimehist)


logch1TrueTimehist = np.log(ch1TrueTimehist)
idx = logch1TrueTimehist != -np.inf
logch1TrueTimehist = logch1TrueTimehist[idx]
binsc = bins[0:-1]+0.5
binscidx = binsc[idx]



x = binscidx #normalized deltat/interval difference

y = logch1TrueTimehist #normalized for probability instead of counts

mfit,bfit = np.polyfit(x , y ,1)

############## paper MLE

lamvalue = np.array([0.1 ,-mfit]) # -mfit bc lambda negative in equation
i = len(lamvalue)
while (np.abs(lamvalue[i-1]-lamvalue[i-2])/lamvalue[i-2]) > 0.005:

    lamiteration =( np.exp(-lamvalue[i-1]*x[0]) - np.exp(-lamvalue[i-1]*x[-1]) ) \
    / ( (avgexp_unitless - x[0]) * np.exp(-lamvalue[i-1]*x[0]) \
     - (avgexp_unitless - x[-1]) * np.exp(-lamvalue[i-1]*x[-1]) )
        
    lamvalue = np.append(lamvalue,lamiteration)
    i = i + 1
    
#################################

#################### poisson sim
lambda1 = 1/avgexp_unitless # 1/ (time per photon)

lambda2 = -mfit # slope naive line fit = -m 

lambda3 = np.exp(bfit)  # = intercept naive line fit = e^b 

lambda4 = lamvalue[-1] # iteration method from paper considering weighted regression


lambdaall = np.array([lambda1, lambda2, lambda3, lambda4])


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

axis.set_xlabel('delta_t/delta',**font)
axis.set_ylabel('pmf',**font)


figure2, axis = plt.subplots(1, 1,constrained_layout=True)

axis.scatter(binsc , exp1, s=100, c='r', marker="o", label='interval count')
axis.scatter(binsc , exp2, s=100, c='b', marker="o", label='interval count')
axis.scatter(binsc , exp3, s=200, c='g', marker="*", label='interval count')
axis.scatter(binsc , exp4, s=100, c='purple', marker="X", label='interval count')
axis.bar(binsc , ch1TrueTimehist, alpha = 1, fill = False, label='interval count')
#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel('delta_t/delta',**font)
axis.set_ylabel('pmf',**font)






plt.grid()
plt.show()



