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
        
        span = (o.ub - o.lb) * sigma
        
        while True:
            x_new = x + rng.uniform(-span, span, size=o.lb.shape)
            x_new = np.clip(x_new, o.lb, o.ub)  
            val = o(x_new)
            if val < current_val:
                x = x_new
                current_val = val
    except StopIteration:
        pass
def solis_wets(o, rng):
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
                x_plus = x + b + d
                f_plus = o(x_plus)
                
                if f_plus < fx:
                    x = np.clip(x_plus, o.lb, o.ub)
                    fx = f_plus
                    b = 0.2 * b + 0.4 * d
                    succ += 1
                    fail = 0
                else:
                    x_minus = x - b - d
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
                
                if succ >= 5:
                    sigma *= 2.0
                    succ = 0
                elif fail >= 3:
                    sigma *= 0.5
                    fail = 0
    except StopIteration:
        pass