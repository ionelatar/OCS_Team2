import numpy as np
from oracle import Oracle
from cec05 import make_cec05
from rs import localized_random_search

D = 10
MaxFE = 10000 * D  
F_all, (lo, hi) = make_cec05(D)

F_test = {'F1': F_all['F1'], 'F3': F_all['F3']}

print("\n Table 2 for D = 10, MaxFE = 10000 * D")
sigmas = [0.01, 0.05, 0.1, 0.3]

for fname, f in F_test.items():
    row_medians = []
    for sigma in sigmas:
        errors = []
        # lambda function for fixing sigma parameter
        alg = lambda o, r, s=sigma: localized_random_search(o, r, sigma=s)
        
        for seed in range(1, 26):
            o = Oracle(f, [lo]*D, [hi]*D, max_fe=MaxFE)
            rng = np.random.default_rng(seed)
            alg(o, rng)
            errors.append(o.best_f)
        
        # compute median of errors
        med = np.median(errors)
        row_medians.append(f"{med:.3g}")
        
    print(f"{fname} for sigmas: {sigmas}: {row_medians}")


print("\nTurning Point for  F1")
sigma_curent = 0.001
f1 = F_all['F1']
best_median = np.inf

while True:
    errors = []
    alg = lambda o, r, s=sigma_curent: localized_random_search(o, r, sigma=s)
    
    for seed in range(1, 26):
        o = Oracle(f1, [lo]*D, [hi]*D, max_fe=MaxFE)
        rng = np.random.default_rng(seed)
        alg(o, rng)
        errors.append(o.best_f)
        
    current_median = np.median(errors)
    print(f"Sigma = {sigma_curent} -> Median for errors = {current_median}")
    
    # if the median erros is greater than best median we have to stop
    if current_median > best_median:
        print("\nTurning point found for F1")
        break
        
    best_median = current_median
    sigma_curent = sigma_curent / 3.0