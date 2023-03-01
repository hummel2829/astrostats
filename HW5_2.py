

import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
from math import factorial

filename = 'xray.txt'
xraydata = np.loadtxt(filename)
xraydata = xraydata.astype(int)

rows =[*range(0,xraydata.shape[0])]

dates = [datetime.datetime(*x) for x in xraydata]

print(dates[1]-dates[0])

cc = np.diff(dates)
timediff = [x.seconds for x in cc]


plt.hist(timediff)




