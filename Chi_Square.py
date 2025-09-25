from scipy.stats import chi2_contingency
from scipy.stats import chi2

table=[[200,150,50],
       [250,300,50]]
print(table)
stat, p, dof, expected = chi2_contingency(table, correction=False)
print('dof = %d' %dof)
print(expected)

prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability = %.3f\ncritical = %.3f\nstat = %.3f' %(prob, critical, stat))

if abs(stat) >= critical:
    print('Dependent(H1)')
else:
    print('Independent(H0)')
print()
alpha = 1.0 - prob
print('significance = %.3f\np = %.5f' %(alpha,p))
if p <= alpha:
    print('Dependent(H1)')
else:
    print('Independent(H0)')