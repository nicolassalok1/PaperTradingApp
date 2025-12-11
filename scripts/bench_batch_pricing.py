import time
import numpy as np
from app.model.options.engines.black_scholes import price_call_bs

S = np.linspace(50, 150, 5000)
K = 100
r = 0.01
sigma = 0.2
T = 1.0


def scalar():
    return [price_call_bs(float(s), K, r, sigma, T) for s in S]


def vectorized():
    from app.model.options.engines.black_scholes import price_call_bs_vectorized

    return price_call_bs_vectorized(S, K, r, sigma, T)


t0 = time.time()
scalar()
print("scalar:", time.time() - t0)

t0 = time.time()
vectorized()
print("vectorized:", time.time() - t0)
