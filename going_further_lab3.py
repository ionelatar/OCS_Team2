import numpy as np
from oracle import Oracle
from cec05 import make_cec05
from rs import solis_wets

print(" Going Further: Solis-Wets Parameters")
configs = [(5, 3), (3, 1), (10, 5)]

for D in (5, 10):
    MaxFE = 10000 * D
    F_all, (lo, hi) = make_cec05(D)
    F_test = {'F1': F_all['F1'], 'F5': F_all['F5']}
    
    print(f"\nDimensiune D = {D}")
    for fname, f in F_test.items():
        row_results = []
        for (S, F) in configs:
            errors = []
            # s and f for algorithm parameters
            alg = lambda o, r, s_val=S, f_val=F: solis_wets(o, r, S=s_val, F=f_val)
            
            for seed in range(1, 26):
                o = Oracle(f, [lo]*D, [hi]*D, max_fe=MaxFE)
                rng = np.random.default_rng(seed)
                alg(o, rng)
                errors.append(o.best_f)
            
            med = np.median(errors)
            row_results.append(f"{med:.3e}")
            
        print(f"  {fname}: (S=5,F=3): {row_results[0]} | (S=3,F=1): {row_results[1]} | (S=10,F=5): {row_results[2]}")