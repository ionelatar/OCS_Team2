import numpy as np, pandas as pd
from oracle import Oracle
from scipy.optimize import minimize
from cec05 import make_cec05, rastrigin  #your code for CEC2005 or the
from rs import pure_random_search, localized_random_search #implementations of RS an PRS
from rs import localized_random_search_restarts, solis_wets
import time
from scipy.stats import friedmanchisquare, wilcoxon
from NM import nm

ALGS = {
    'PRS': pure_random_search,
    'LRS': localized_random_search, 
    'LRS with restarts': localized_random_search_restarts,
    'SW': solis_wets
}



def run_one(alg, f, D, lo, hi, seed):
    o = Oracle(f, [lo] * D, [hi] * D, max_fe=10000 * D)
    ALGS[alg](o, np.random.default_rng(seed))
    return o.best_f, o.fe, o.trace

if __name__ == '__main__':
    start = time.perf_counter()
    rows = []
    """
    for D in (5, 10):
        F, (lo, hi) = make_cec05(D)
        for fname, f in F.items():
            for alg in ALGS:
                for seed in range(1, 26):
                    err, fe, _ = run_one(alg, f, D, lo, hi, seed)
                    rows.append((alg, fname, D, seed, err, fe))
    df = pd.DataFrame(rows, columns=['alg', 'func', 'D', 'run', 'error', 'fe'])
    df.to_csv('l3_results.csv', index=False)
    """
    lo, hi = -5.12, 5.12
    for D in (2, 5):
        for alg in ALGS:
            for seed in range(1, 26):
                err, fe, _ = run_one(alg, rastrigin, D, lo, hi, seed)
                rows.append((alg, 'Rastrigin', D, seed, err, fe))
                
    df = pd.DataFrame(rows, columns=['alg', 'func', 'D', 'run', 'error', 'fe'])
    df.to_csv('l3_LRS_restart_results.csv', index=False)


    elapsed = time.perf_counter() - start
    print(f"Runtime: {elapsed:.2f} s")

    