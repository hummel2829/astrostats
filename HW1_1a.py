import numpy as np
import matplotlib.pyplot as plt

X = []

n = 100
draws = 10000

# Can be written w/o loop using NumPy functions.
for i in range(0, 10000):
    C = np.random.randn(1, draws)
    Xbar = np.mean(C)
    X = np.append(X,Xbar)

# For future consideration: Never use default bins for hist. Here distribution
# is symmetric, so use symmetric x limits.
# Labels!
plt.hist(X)
