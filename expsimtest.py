
import matplotlib.pyplot as plt
import numpy as np

lambdatest = 0.1133
#lambdatest = 1

expsimdata =np.sort( np.random.exponential(lambdatest , 10000))
expsimdata = np.flip(expsimdata)

'''
bins = np.arange(0,10,0.1)
plt.figure(1)
exphist, bins, ign = plt.hist(expsimdata,bins)


logexphist = np.log(exphist/np.sum(exphist))

idx = logexphist != -np.inf

logexphist = logexphist[idx]

binsc = bins[0:-1] + np.diff(bins)[0]

binscidx = binsc[idx]



y = np.exp(logexphist)
x = binscidx
'''

y = expsimdata
x = ( np.log(y) - np.log(lambdatest) )/(-lambdatest)
#ynew = lambdatest*np.exp(-lambdatest*x)

#logy = np.log(y)

#idxfit = logy != -np.inf
#x = binscidx[idxfit]
#y = logexphist[idxfit]

#mwolf = (np.sum(y) * np.sum(x * y * np.log(y)) - np.sum(x*y) * np.sum(y*np.log(y))) \
 #   / ( np.sum(y) * np.sum(x * x * y) - np.sum(x * y)**2 )


#bwolf = ( np.sum(x * x * y) * np.sum(y*np.log(y)) - np.sum(x * y) * np.sum(x * y * np.log(y)) ) \
  #  / ( np.sum(y) * np.sum(x * x * y) - np.sum(x * y)**2 )


avgexp_unitless = np.mean(y)

mfit,bfit = np.polyfit(x , np.log(y) ,1)

############## paper MLE


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

mj = (np.sum(np.log(y)) * np.sum(x * np.log(y) * np.log(y)) - np.sum(x*np.log(y)) * np.sum(np.log(y)*np.log(y))) \
    / ( np.sum(np.log(y)) * np.sum(x * x * np.log(y)) - np.sum(x * np.log(y))**2 )


bj = ( np.sum(x * x * np.log(y)) * np.sum(np.log(y)*np.log(y)) - np.sum(x * np.log(y)) * np.sum(x * np.log(y) * np.log(y)) ) \
    / ( np.sum(np.log(y)) * np.sum(x * x * np.log(y)) - np.sum(x * np.log(y))**2 )


##############################################

#################### poisson sim
lambda1 = (avgexp_unitless) # 1/ (time per photon)

lambda2 = -mfit # slope naive line fit = -m 

lambda3 = np.exp(bfit)  # = intercept naive line fit = e^b 

lambda4 = lamvalue[-1] # iteration method from paper considering weighted regression

lambda5 = mj

lambda6 = np.exp(bj)

lambdaall = np.array([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6])



#mj = (np.sum(np.log(y)) * np.sum(x * np.log(y) * np.log(y)) - np.sum(x*np.log(y)) * np.sum(np.log(y)*np.log(y))) \
#    / ( np.sum(np.log(y)) * np.sum(x * x * np.log(y)) - np.sum(x * np.log(y))**2 )


#bj = ( np.sum(x * x * np.log(y)) * np.sum(np.log(y)*np.log(y)) - np.sum(x * np.log(y)) * np.sum(x * np.log(y) * np.log(y)) ) \
#    / ( np.sum(np.log(y)) * np.sum(x * x * np.log(y)) - np.sum(x * np.log(y))**2 )


figure2, axis = plt.subplots(1, 1,constrained_layout=True)

#axis.plot(x , np.log(y),  c='r', marker="o", label='1/mean fit,  lambda1 = %1.5f' %lambda1)
#axis.plot(x , mwolf*x + bwolf,  c='purple', marker="o", ms = 8, label='1/mean fit,  lambda1 = %1.5f' %lambda1)
axis.plot(x , mfit*x + bfit,  c='brown', marker="o", label='1/mean fit,  lambda1 = %1.5f' %lambda1)
axis.plot(x , mj*x + bj,  c='blue', marker="o", label='1/mean fit,  lambda1 = %1.5f' %lambda1)
axis.plot(x , np.log(y)*(np.log(y) - mj*x - bj)**2,  c='orange', marker="o", label='1/mean fit,  lambda1 = %1.5f' %lambda1)
axis.plot(x , np.log(y)*(np.log(y) - mfit*x - bfit)**2,  c='green', marker="o", label='1/mean fit,  lambda1 = %1.5f' %lambda1)
axis.scatter(x , np.log(y))


#axis.plot(binsc , np.log(exp2),  c='b', marker="o", label='slope of lin reg, lambda2 = %1.5f' %lambda2)
#axis.plot(binsc , np.log(exp3),  c='g', marker="*", label='intercept of lin reg, lambda3 = %1.5f' %lambda3)
#axis.plot(binsc , np.log(exp4),  c='purple', marker="X", label='iteration of lambda, lambda4 = %1.5f' %lambda4)
#axis.plot(binsc , np.log(ch1TrueTimehist), label='interval count')
#axis.legend(loc='upper right',fontsize = 25)

font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('Simulated exp distributions error' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font)
axis.set_ylabel('log(exp fit) - log(data)',**font)



figure3, axis = plt.subplots(1, 1,constrained_layout=True)

axis.plot(x , np.log(y) - (mfit*x + bfit),  c='brown', marker="o", ms = 10, label='polyfit')
axis.plot(x , np.log(y) - (mj*x + bj),  c='blue', marker="o", label = 'john')

#axis.scatter(x , np.log(y))


font = {'fontname' : 'Times New Roman' , 'size' : 25}
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
#plt.ticklabel_format(axis='both', style='plain')

axis.set_title('Simulated exp distributions error' ,**font)

axis.set_xlabel(r'$\Delta$t/$\delta$',**font)
axis.set_ylabel('log(exp fit) - log(data)',**font)


plt.grid()
plt.show()



