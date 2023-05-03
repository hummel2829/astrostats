import matplotlib.pyplot as plt
import numpy as np
from math import ceil

lambda1 = 1.1335066003515445e-08 # sum of counts/total time  photon/picosecond

lambda2 = 1.0894839325117185e-08 # slope naive line fit = -m 

lambda3 = 0.1064427069887388 # = intercept naive line fit = e^b 

measurements = 10000

#poisson sim


nanobysec = (1e9) #nanoseconds in sec, convert photon/pico to photon/milisec
lam =  [lambda1*nanobysec, lambda2*nanobysec, lambda3]
pdata = np.random.poisson(lam,(measurements,len(lam)))

pdata1 = pdata[:,0]
pd1 = np.reshape(pdata1,(100,100))

pdata2 = pdata[:,1]
pd2 = np.reshape(pdata2,(100,100))

pdata3 = pdata[:,2]
pd3 = np.reshape(pdata3,(100,100))

'''
pdbootmean = np.average(np.random.choice(pdata,[10000,20]),axis=1)

#Hnull is pdbootmean == stdpdata
#Halt is pdbootmean not= stdpdata
#alpha = 99.5, Z table value 2.58, significance level 0.01

zpd = (np.mean(pdbootmean) - np.std(pdata)**2)/(np.std(pdbootmean)/(10000**0.5))

# -2.58 < zpd=1.34 < 2.58 so cannot reject Hnull

pdmeanhist = np.histogram(pdbootmean, bins = 100)

pstd = np.std(pdata)
'''

w1 = np.ones(len(pdata1))/len(pdata1)
pdata1hist = np.histogram(pdata1,bins=np.max(pd1)+1, weights = w1)

w2 = np.ones(len(pdata2))/len(pdata2)
pdata2hist = np.histogram(pdata2,bins=np.max(pd2)+1, weights = w2)

w3 = np.ones(len(pdata3))/len(pdata3)
pdata3hist = np.histogram(pdata3,bins=np.max(pd3)+1, weights = w3)







figure1, axis = plt.subplots(1, 1,constrained_layout=True)

x1hist = pdata1hist[1][0:-1] + (pdata1hist[1][0:-1] - pdata1hist[1][0:-1])/2

axis.scatter(x1hist, pdata1hist[0], s=50, c='r', marker="o", label='interval count')
#axis.scatter(pdata2hist[1][0:-1], pdata2hist[0], s=50, c='b', marker="o", label='interval count')
#axis.scatter(pdata3hist[1][0:-1], pdata3hist[0], s=100, c='g', marker="*", label='interval count')


#plt.xlim(0,0.5)


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel('success',**font)
axis.set_ylabel('pdf',**font)


figure2, axis = plt.subplots(1, 1,constrained_layout=True)

x1hist = pdata1hist[1][0:-1] + (pdata1hist[1][0:-1] - pdata1hist[1][0:-1])/2
x2hist = pdata2hist[1][0:-1] + (pdata2hist[1][0:-1] - pdata2hist[1][0:-1])/2

axis.bar(x1hist, pdata1hist[0],alpha = 0.5, label='interval count')
axis.bar(pdata2hist[1][0:-1], pdata2hist[0],alpha = 0.5, label='interval count')
#axis.scatter(pdata2hist[1][0:-1], pdata2hist[0], s=50, c='b', marker="o", label='interval count')
#axis.scatter(pdata3hist[1][0:-1], pdata3hist[0], s=100, c='g', marker="*", label='interval count')


#plt.xlim(0,0.5)


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_xlabel('success',**font)
axis.set_ylabel('pdf',**font)





plt.grid()
plt.show()









