from scipy.stats import chi2_contingency, chi2

table = [[209, 280],
         [225, 248]]

stat, p, dof, expected = chi2_contingency(table)

print(f'dof = {dof}')
print('Expected:\n', expected)

# 유의수준
alpha = 0.05
critical = chi2.ppf(1 - alpha, dof)  # 95% 신뢰수준 → upper 5% critical value
print(f'critical = {critical:.3f}, stat = {stat:.3f}')

if stat >= critical:
    print("Dependent (H1)")
else:
    print("Independent (H0)")

print(f'significance = {alpha:.3f}, p = {p:.3f}')
if p <= alpha:
    print("Dependent (H1)")
else:
    print("Independent (H0)")
