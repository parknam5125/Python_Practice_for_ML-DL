import numpy as np
from scipy import stats

A = np.array([45, 38, 52, 48, 25, 39, 51, 46, 55, 46])
B = np.array([34, 22, 15, 27, 37, 41, 24, 19, 26, 36])

N = 10

var_A = A.var(ddof=1)
var_B = B.var(ddof=1)

s = np.sqrt((var_A + var_B)/2)
t = (A.mean() - B.mean())/(s*np.sqrt(2/N))

df = 2*N - 2

p = 1 - stats.t.cdf(t,df=df)

print("t = " + str(t))
print("p = " + str(2*p))

t2, p2 = stats.ttest_ind(A,B)

print("t = " + str(t2))
print("p = " + str(p2))