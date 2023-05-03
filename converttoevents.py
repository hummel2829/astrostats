
import matplotlib.pyplot as plt
import numpy as np
from math import ceil



filename = r'D:\Python\picoquant\antibunch\NV-Center_for_Antibunching_1.out'
NVdata = np.loadtxt(filename, delimiter = ',', usecols = (0,1,2), skiprows = 1)
NVdata = NVdata.astype(np.int64)

ch1count = np.count_nonzero(NVdata[:,0] == 1)
ch1and99count = np.count_nonzero(NVdata[:,0] != 2)


ch1index = np.nonzero(NVdata[:,0] == 1)
ch1and99index = np.nonzero(NVdata[:,0] != 2)

NVch1TimeTag = NVdata[ch1index,1]
NVch1TrueTime = NVdata[ch1index,2]

NVch1TrueTimet1stdiff = np.diff(NVch1TrueTime)
x1stdiff = np.arange(0,NVch1TrueTimet1stdiff.shape[1])







ch1and99count = np.count_nonzero(NVdata[:,0] != 2)


ch1and99index = np.nonzero(NVdata[:,0] != 2)

NVch1and99TimeTag = NVdata[ch1and99index,1]
NVch1and99TrueTime = NVdata[ch1and99index,2]

newNVch1and99TrueTime = np.copy(NVch1and99TrueTime)
newNVch1and99TrueTime.resize([ceil(NVch1and99TrueTime[0].shape[0]/100),100])
NV0 = np.where(newNVch1and99TrueTime != -1, newNVch1and99TrueTime, 0)
NV01 = np.where(NV0 == 0, NV0, 1)
# approx 1513063887ps per row
NV01counts = np.sum(NV01,axis=1)





measurements = NV01.shape[0]

#poisson sim

photonrate = 3e8 #photon/second
timeinterval = 1/(3e8) #seconds
lam = photonrate*timeinterval  
pdata = np.random.poisson(lam,measurements)

#pd = np.reshape(pdata,(NV01.shape[0],NV01.shape[1]))


figure1, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')

#barindex = np.nonzero(ch1TrueTimehist[1] < 1e9)
#axis.scatter(ch1TrueTimehist[1][0:barindex[0][-1]] , ch1TrueTimehist[0][0:barindex[0][-1]]/np.sum(ch1TrueTimehist[0][0:barindex[0][-1]]) , s=50, c='r', marker="o", label='interval count')


axis.hist(NV01counts)
axis.hist(pdata, color = 'red')

axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

axis.set_title('logy = %1.3e x' %m + ' %1.3f' %b + '  ,  avg photonrate =  %1.3e' %photonrate ,**font)

axis.set_ylabel('pdf',**font)
axis.set_xlabel('time between photon counts (picoseconds)',**font)



plt.grid()
plt.show()



















