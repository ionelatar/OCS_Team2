import numpy as np

def pure_random_search(o, rng):
    try:
        while True:
            x = rng.uniform(o.lb, o.ub)
            o(x)  
    except StopIteration:
        pass

def localized_random_search(o, rng, sigma=0.1):
    try:
        x = rng.uniform(o.lb, o.ub)
        current_val = o(x)
        step_std = (np.asarray(o.ub) - np.asarray(o.lb)) * sigma
        while True:
            x_new = np.clip(x + rng.normal(0.0, step_std, size=o.lb.shape), o.lb, o.ub)
            val = o(x_new)
            if val < current_val:
                x = x_new
                current_val = val
    except StopIteration:
        pass
        
def solis_wets(o, rng, S=5, F=3):
    D = len(o.lb)
    span = o.ub - o.lb
    try:
        while True:
            x = rng.uniform(o.lb, o.ub)
            fx = o(x)
            b = np.zeros(D)
            sigma = 0.2 * span
            succ, fail = 0, 0
            
            while np.max(sigma) >= 1e-8:
                d = rng.normal(0.0, sigma, size=D)
                x_plus = np.clip(x + b + d, o.lb, o.ub)
                f_plus = o(x_plus)
                
                if f_plus < fx:
                    x = np.clip(x_plus, o.lb, o.ub)
                    fx = f_plus
                    b = 0.2 * b + 0.4 * d
                    succ += 1
                    fail = 0
                else:
                    x_minus = np.clip(x - b - d, o.lb, o.ub)
                    f_minus = o(x_minus)
                    if f_minus < fx:
                        x = np.clip(x_minus, o.lb, o.ub)
                        fx = f_minus
                        b = b - 0.4 * d
                        succ += 1
                        fail = 0
                    else:
                        b = 0.5 * b
                        succ = 0
                        fail += 1
                
                if succ >= S:
                    sigma *= 2.0
                    succ = 0
                elif fail >= F:
                    sigma *= 0.5
                    fail = 0
    except StopIteration:
        pass

def localized_random_search_restarts(o, rng, sigma=0.1, max_stall=100):
    lo = o.lb
    hi = o.ub
    D = len(lo)
    step = sigma * (hi - lo)

    try:
        x = rng.uniform(lo, hi)
        fx = o(x)
        stall = 0

        while o.fe < o.max_fe:
            cand = x + rng.normal(0.0, step, size=D)
            fcand = o(cand)

            if fcand < fx:
                x = cand
                fx = fcand
                stall = 0
            else:
                stall += 1

            if stall >= max_stall and o.fe < o.max_fe:
                x = rng.uniform(lo, hi)
                fx = o(x)
                stall = 0
    except StopIteration:
        pass