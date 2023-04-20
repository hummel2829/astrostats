
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


















