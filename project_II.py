
import matplotlib.pyplot as plt
import numpy as np

filename = r'D:\Python\picoquant\antibunch\NV-Center_for_Antibunching_1.out'
NVdata = np.loadtxt(filename, delimiter = ',', usecols = (0,1,2), skiprows = 1)
NVdata = NVdata.astype(np.int64)


ch1count = np.count_nonzero(NVdata[:,0] == 1)


ch1index = np.nonzero(NVdata[:,0] == 1)

NVch1 = NVdata[ch1index,1]

NVch1t1stdiff = np.diff(NVch1)
x1stdiff = np.arange(0,NVch1t1stdiff.shape[1])

ch1hist = np.histogram(NVch1t1stdiff,bins=100)
xx = np.arange(0,100)

ovr = 1000 #number of positions away from initial poisition for product(i.e. avr = 3 is product of 1st & 4th, 2nd & 5th etc)
movingprodch1 = np.zeros(NVch1t1stdiff.shape[1])
for i in range (0,NVch1t1stdiff.shape[1]-ovr):
    movingprodch1[i] = NVch1t1stdiff[0,i]*NVch1t1stdiff[0,i+ovr]
    
pearcorrcoff = np.sum(movingprodch1)/(NVch1t1stdiff.shape[1]*np.std(NVch1t1stdiff)**2)
print(pearcorrcoff)




np.mean(NVch1) < np.std(NVch1)**2 #True mean < var
avgphotonratech1 = NVch1.shape[1]/NVch1[0,-1] #counts/total time (5.6675e-9 counts/second)



figure1, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
axis.scatter(x1stdiff, NVch1t1stdiff, s=25, c='r', marker="o", label='interval count')


#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='y', style='plain')

#axis.set_title('avg interval = %1.3f' %np.mean(minbetween) + ' , std interval = %1.3f' %np.std(minbetween) + ' , z score = %1.5f < 2.33' %zinterval ,**font)
axis.set_ylabel('time between photon count (picoseconds)',**font)
axis.set_xlabel('interval number',**font)


figure2, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x,minbhist[0], c='r', marker="o", label='poisson light')
#axis.scatter(xx, ch1hist[0], s=50, c='r', marker="o", label='interval count')
barindex = np.nonzero(ch1hist[1] < 1e9)

axis.bar(ch1hist[1][0:barindex[0][-1]],ch1hist[0][0:barindex[0][-1]],width = 1000000, color = 'blue')

#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ticklabel_format(axis='both', style='plain')

#axis.set_title('avg interval = %1.3f' %np.mean(minbetween) + ' , std interval = %1.3f' %np.std(minbetween) + ' , z score = %1.5f < 2.33' %zinterval ,**font)
axis.set_ylabel('counts',**font)
axis.set_xlabel('time between photon counts (picoseconds',**font)









plt.grid()
plt.show()








