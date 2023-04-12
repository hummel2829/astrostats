
import matplotlib.pyplot as plt
import numpy as np

filename = r'D:\Python\picoquant\antibunch\NV-Center_for_Antibunching_1.out'
NVdata = np.loadtxt(filename, delimiter = ',', usecols = (0,1,2), skiprows = 1)
NVdata = NVdata.astype(np.int64)


ch1count = np.count_nonzero(NVdata[:,0] == 1)


ch1index = np.nonzero(NVdata[:,0] == 1)

NVch1 = NVdata[ch1index,1]

