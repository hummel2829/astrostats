
import matplotlib.pyplot as plt
import numpy as np
    from scipy.special import factorial

filename = r'D:\Python\picoquant\antibunch\NV-Center_for_Antibunching_1.out'
NVdata = np.loadtxt(filename, delimiter = ',', usecols = (0,1,2), skiprows = 1)
NVdata = NVdata.astype(np.int64)


ch1count = np.count_nonzero(NVdata[:,0] == 1)


ch1index = np.nonzero(NVdata[:,0] == 1)

NVch1TimeTag = NVdata[ch1index,1]
NVch1TrueTime = NVdata[ch1index,2]
NVch1TrueTime = NVch1TrueTime[0]

NVch1TrueTimet1stdiff = np.diff(NVch1TrueTime)
x1stdiff = np.arange(0,NVch1TrueTimet1stdiff.shape[0])

photonpersecond = len(NVch1TrueTime) / NVch1TrueTime[-1] # number photon / total measurement time (picoseconds)

bins=np.arange(0,10,0.1)

#intervaldiff = 1e7
avgintervaldiff = np.average(np.abs(np.diff(NVch1TrueTimet1stdiff)))


ch1TrueTimehist , histbins = np.histogram(NVch1TrueTimet1stdiff/avgintervaldiff,bins=bins)

#avgexp_unitless = (np.sum(NVch1TrueTimet1stdiff)/avgintervaldiff) / len(NVch1TrueTimet1stdiff)

ch1TrueTimehist = ch1TrueTimehist/np.sum(ch1TrueTimehist)


idx = ch1TrueTimehist != 0

ch1TrueTimehist = ch1TrueTimehist[idx]

logch1TrueTimehist = np.log(ch1TrueTimehist)
#idx = logch1TrueTimehist != -np.inf
#logch1TrueTimehist = logch1TrueTimehist[idx]
binsc = bins[0:-1]+ np.diff(bins)[0]/2
binscidx = binsc[idx]

#plt.plot(expsimdata)
#plt.show

y = ch1TrueTimehist
x = binscidx


avgexp_unitless = np.sum(y*x) #(prob * bin)

mfit,bfit = np.polyfit(x , np.log(y) ,1)


lamold = -mfit # -mfit bc lambda negative in equation

while True:
    print(lamold)

    lamnew =( np.exp(-lamold * x[0]) - np.exp(-lamold * x[-1]) ) \
    / ( (avgexp_unitless - x[0]) * np.exp(-lamold * x[0]) \
     - (avgexp_unitless - x[-1]) * np.exp(-lamold * x[-1]) )
        
    if (np.abs(lamnew-lamold)/lamold) < 0.00001:
        break
    lamold = lamnew

lamiteration = lamold


x0 = x[0]

while True:
    #print(x0)
    lamnew = 1/ (avgexp_unitless - x0)
    
    if (np.abs(lamnew-lamold)/lamold) < 0.00001:
        break
    x0 = x0*0.99
    lamold = lamnew

lamsimpleiteration = lamold
xoptimal = x0

print(lamsimpleiteration, xoptimal)    


mj = (np.sum(np.log(y)) * np.sum(x * np.log(y) * np.log(y)) - np.sum(x*np.log(y)) * np.sum(np.log(y)*np.log(y))) \
    / ( np.sum(np.log(y)) * np.sum(x * x * np.log(y)) - np.sum(x * np.log(y))**2 )


bj = ( np.sum(x * x * np.log(y)) * np.sum(np.log(y)*np.log(y)) - np.sum(x * np.log(y)) * np.sum(x * np.log(y) * np.log(y)) ) \
    / ( np.sum(np.log(y)) * np.sum(x * x * np.log(y)) - np.sum(x * np.log(y))**2 )


##############################################

################### lambda approximations
lambda1 = 1/(avgexp_unitless) # 1/ (time per photon)

lambda2 = -mfit # slope naive line fit = -m 

lambda3 = np.exp(bfit)  # = intercept naive line fit = e^b 

lambda4 = lamiteration # iteration method from paper 

lambda5 = lamsimpleiteration #iteration varying x0, location of first histogram bar

lambda6 = -mj

lambda7 = np.exp(bj)

lambdaall = np.array([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7])


###########################  exp distribution sim for theoretical comparison

'''

exp1simdata =np.sort( np.random.exponential(1/lambda1 , 10000))
exp1simdata = np.flip(exp1simdata)

exp2simdata =np.sort( np.random.exponential(1/lambda2 , 10000))
exp2simdata = np.flip(exp2simdata)

exp3simdata =np.sort( np.random.exponential(1/lambda3 , 10000))
exp3simdata = np.flip(exp3simdata)

exp4simdata =np.sort( np.random.exponential(1/lambda4 , 10000))
exp4simdata = np.flip(exp4simdata)

exp5simdata =np.sort( np.random.exponential(1/lambda5 , 10000))
exp5simdata = np.flip(exp5simdata)

exp6simdata =np.sort( np.random.exponential(1/lambda6 , 10000))
exp6simdata = np.flip(exp6simdata)

exp7simdata =np.sort( np.random.exponential(1/lambda7 , 10000))
exp7simdata = np.flip(exp7simdata)

exp1hist, exp1bins, ign = plt.hist(exp1simdata, bins=bins)
exp2hist, exp2bins, ign = plt.hist(exp2simdata, bins=bins)
exp3hist, exp3bins, ign = plt.hist(exp3simdata, bins=bins)
exp4hist, exp4bins, ign = plt.hist(exp4simdata, bins=bins)
exp5hist, exp5bins, ign = plt.hist(exp5simdata, bins=bins)
exp6hist, exp6bins, ign = plt.hist(exp6simdata, bins=bins)
exp7hist, exp7bins, ign = plt.hist(exp7simdata, bins=bins)

'''

###########################  exp equation sim for theoretical comparison


exp1 = lambda1*np.exp(-lambda1*x)

exp2 = lambda2*np.exp(-lambda2*x)

exp3 = lambda3*np.exp(-lambda3*x)

exp4 = lambda4*np.exp(-lambda4*x)

exp5 = lambda5*np.exp(-lambda5*x)

exp6 = lambda6*np.exp(-lambda6*x)

exp7 = lambda7*np.exp(-lambda7*x)

exp1 = exp1/np.sum(exp1)
exp2 = exp2/np.sum(exp2)
exp3 = exp3/np.sum(exp3)
exp4 = exp4/np.sum(exp4)
exp5 = exp5/np.sum(exp5)
exp6 = exp6/np.sum(exp6)
exp7 = exp7/np.sum(exp7)

########################### poisson sim

#k = np.arange(0,10,1)
#ptest = (lambda1**k * np.exp(-lambda1))/factorial(k)


pdata = np.random.poisson(lambdaall,(10000,len(lambdaall)))

pdata1 = pdata[:,0]

pdata2 = pdata[:,1]

pdata3 = pdata[:,2]

pdata4 = pdata[:,3]

pdata5 = pdata[:,4]

pdata6 = pdata[:,5]

pdata7 = pdata[:,6]

pdata1hist , pbins1, ign = plt.hist(pdata1, bins= bins)
pdata2hist , pbins2, ign = plt.hist(pdata2, bins= bins)
pdata3hist , pbins3, ign = plt.hist(pdata3, bins= bins)
pdata4hist , pbins4, ign = plt.hist(pdata4, bins= bins)
pdata5hist , pbins5, ign = plt.hist(pdata5, bins= bins)
pdata6hist , pbins6, ign = plt.hist(pdata6, bins= bins)
pdata7hist , pbins7, ign = plt.hist(pdata7, bins= bins)

pdata1hist = pdata1hist/np.sum(pdata1hist)
pdata2hist = pdata2hist/np.sum(pdata2hist)
pdata3hist = pdata3hist/np.sum(pdata3hist)
pdata4hist = pdata4hist/np.sum(pdata4hist)
pdata5hist = pdata5hist/np.sum(pdata5hist)
pdata6hist = pdata6hist/np.sum(pdata6hist)
pdata7hist = pdata7hist/np.sum(pdata7hist)


pdataallhist = np.stack([pdata1hist, pdata2hist, pdata3hist, pdata4hist, pdata5hist, pdata6hist, pdata7hist],axis = 0)

muall = np.zeros(pdataallhist.shape[0])
varall = np.zeros(pdataallhist.shape[0])

for i in range(pdataallhist.shape[0]):
    muall[i] = np.sum(pdataallhist[i]*x)
    varall[i] = np.sum(pdataallhist[i]*(x-muall[i])**2)

pdata1histboot = np.random.choice(pdata1hist, size = (10000, len(pdata1hist)))
mu1boot = np.sum(pdata1histboot*x, axis = 1)
mu1boot_T = mu1boot[:,np.newaxis] #reshapes so x-mu[i] is valid
x_boot = np.resize(x_boot, (len(mu1boot),len(x)))
#########################
######################3## DOUBLE CHECK
var1boot = np.sum(pdata1histboot*(x_boot-mu1boot_T)**2)
#DOUBLE CHECK#########################
#############
############
##################### plots

figure1, axis = plt.subplots(1, 1,constrained_layout=True)

axis.plot(x,np.log(y), c = 'black', marker = '*', ms = 20, label = 'NVdata')



#axis.plot(x , np.log(exp1),  c='purple', marker="o", ms = 18, label='lambda1 = %1.5f' %lambda1)
#axis.plot(x , np.log(exp2),  c='brown', marker="o", ms = 16, label='lambda2 = %1.5f' %lambda2)
#axis.plot(x , np.log(exp3),  c='red', marker="o", ms = 14, label='lambda3 = %1.5f' %lambda3)
#axis.plot(x , np.log(exp4),  c='blue', marker="o", ms = 12, label='lambda4 = %1.5f' %lambda4)
axis.plot(x , np.log(exp5),  c='green', marker="o", ms = 10, label='lambda5 = %1.5f' %lambda5)
#axis.plot(x , np.log(exp6),  c='pink', marker="o", ms = 8, label='lambda6 = %1.5f' %lambda6)
#axis.plot(x , np.log(exp7),  c='orange', marker="o", ms = 6, label='lambda7 = %1.5f' %lambda7)


font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(np.arange(0,13,1) , fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('Log plot of exp sim histogram of NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)



figure2, axis = plt.subplots(1, 1,constrained_layout=True)


#axis.plot(x , np.log(y) - np.log(exp1),  c='purple', marker="o", ms = 20, label='lambda1 = %1.5f' %lambda1)
#axis.plot(x , np.log(y) - np.log(exp2),  c='brown', marker="o", ms = 18, label='lambda2 = %1.5f' %lambda2)
#axis.plot(x , np.log(y) - np.log(exp3),  c='red', marker="o", ms = 16, label='lambda3 = %1.5f' %lambda3)
#axis.plot(x , np.log(y) - np.log(exp4),  c='blue', marker="o", ms = 14, label='lambda4 = %1.5f' %lambda4)
axis.plot(x , np.log(y) - np.log(exp5),  c='green', marker="o", ms = 12, label='lambda5 = %1.5f' %lambda5)
#axis.plot(x , np.log(y) - np.log(exp6),  c='pink', marker="o", ms = 10, label='lambda6 = %1.5f' %lambda6)
#axis.plot(x , np.log(y) - np.log(exp7),  c='orange', marker="o", ms = 8, label='lambda7 = %1.5f' %lambda7)


font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)


figure3, axis = plt.subplots(1, 1,constrained_layout=True)

idxp1 = pdata1hist !=0
pdata1histidx = pdata1hist[idxp1]
xp1 = x[idxp1]


idxp2 = pdata2hist !=0
pdata2histidx = pdata2hist[idxp2]
xp2 = x[idxp2]


idxp3 = pdata3hist !=0
pdata3histidx = pdata3hist[idxp3]
xp3 = x[idxp3]


idxp4 = pdata4hist !=0
pdata4histidx = pdata4hist[idxp4]
xp4 = x[idxp4]


idxp5 = pdata5hist !=0
pdata5histidx = pdata5hist[idxp5]
xp5 = x[idxp5]


idxp6 = pdata6hist !=0
pdata6histidx = pdata6hist[idxp6]
xp6 = x[idxp6]


idxp7 = pdata7hist !=0
pdata7histidx = pdata7hist[idxp7]
xp7 = x[idxp7]


axis.plot(xp1 , pdata1histidx,  c='purple', marker="o", ms = 20, label='lambda1 = %1.5f' %lambda1)
axis.plot(xp2 , pdata2histidx,  c='brown', marker="o", ms = 18, label='lambda2 = %1.5f' %lambda2)
axis.plot(xp3 , pdata3histidx,  c='red', marker="o", ms = 16, label='lambda3 = %1.5f' %lambda3)
axis.plot(xp4 , pdata4histidx,  c='blue', marker="o", ms = 14, label='lambda4 = %1.5f' %lambda4)
axis.plot(xp5 , pdata5histidx,  c='green', marker="o", ms = 12, label='lambda5 = %1.5f' %lambda5)
axis.plot(xp6 , pdata6histidx,  c='pink', marker="o", ms = 10, label='lambda6 = %1.5f' %lambda6)
axis.plot(xp7 , pdata7histidx,  c='orange', marker="o", ms = 8, label='lambda7 = %1.5f' %lambda7)


font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)


figure4, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(x , pdata1hist,  color='purple', label='lambda1 = %1.5f' %lambda1)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)

figure5, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(x , pdata2hist,  color='brown', label='lambda2 = %1.5f' %lambda2)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)


figure6, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(x , pdata3hist,  color='red', label='lambda3 = %1.5f' %lambda3)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)


figure7, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(x , pdata4hist,  color='blue', label='lambda4 = %1.5f' %lambda4)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)


figure8, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(x , pdata5hist,  color='green', label='lambda5 = %1.5f' %lambda5)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)


figure9, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(x , pdata6hist,  color='pink', label='lambda6 = %1.5f' %lambda6)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)


figure10, axis = plt.subplots(1, 1,constrained_layout=True)

axis.bar(x , pdata7hist,  color='orange', label='lambda7 = %1.5f' %lambda7)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('diff of Log of exp sim and NVdata' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font) #Delta = interval, delta = avg diff of intervals
axis.set_ylabel('log(NVdata)',**font)

axis.legend(loc='upper right',fontsize = 25)




plt.grid()
plt.show()








