
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

ch1TrueTimehist = np.histogram(NVch1TrueTimet1stdiff,bins=100)
xx = np.arange(0,100)

logch1TrueTimehist = np.log(ch1TrueTimehist[0])

loglinefitindex = np.nonzero(logch1TrueTimehist > -1e9) #used to skip -inf from log(0)

photonrate = np.sum(ch1TrueTimehist[0])/NVch1TrueTime[0][-1]
# photonrate = 1.1335066003515445e-08 photon/picosecond

m,b = np.polyfit(ch1TrueTimehist[1][loglinefitindex[0]] , logch1TrueTimehist[loglinefitindex[0]] ,1)
#m = -1.0894839325117183e-08 , b = 11.189833490594655


'''

figure1, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
axis.scatter(x1stdiff, NVch1TrueTimet1stdiff/1e9, s=25, c='r', marker="o", label='interval count')


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='y', style='plain')

#axis.set_title('avg interval = %1.3f' %np.mean(minbetween) + ' , std interval = %1.3f' %np.std(minbetween) + ' , z score = %1.5f < 2.33' %zinterval ,**font)
axis.set_title('Time intervals recorded between photon counts of NV samples' ,**font)
axis.set_ylabel('time between photon count (x10^9 picoseconds)',**font)
axis.set_xlabel('interval number',**font)




figure2, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
#axis.scatter(xx, ch1hist[0], s=50, c='r', marker="o", label='interval count')
barindex = np.nonzero(ch1TrueTimehist[1] < 1e9)

f = axis.semilogy(ch1TrueTimehist[1][0:barindex[0][-1]],ch1TrueTimehist[0][0:barindex[0][-1]], color = 'blue')
ydata = f.get_ydata(orig = True)
#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('mean of counts = %1.3f' %intervalcountmean + ' , std interval counts = %1.3f' %intervalcountstd + ' ,**font)

axis.set_ylabel('counts',**font)
axis.set_xlabel('time between photon counts (picoseconds',**font)

'''


figure3, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')

barindex = np.nonzero(ch1TrueTimehist[1] < 1e9)


axis.scatter(ch1TrueTimehist[1][0:barindex[0][-1]] , logch1TrueTimehist[0:barindex[0][-1]] , s=50, c='r', marker="o", label='interval count')
axis.plot(ch1TrueTimehist[1][loglinefitindex[0]], b + m * ch1TrueTimehist[1][loglinefitindex[0]], color="k", lw=2.5, label = 'linear fit of ylog');
axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

axis.set_title('logy = %1.3e x' %m + ' %1.3f' %b + '  ,  avg photonrate =  %1.3e' %photonrate ,**font)

axis.set_ylabel('counts',**font)
axis.set_xlabel('time between photon counts (picoseconds)',**font)




plt.grid()
plt.show()

