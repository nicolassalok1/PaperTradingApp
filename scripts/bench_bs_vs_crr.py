import time
import numpy as np

from app.model.options.engines.black_scholes import price_call_bs
from app.model.options.engines.crr import price_call_crr

S0 = 100
K = 100
r = 0.01
sigma = 0.2
T = 1.0

N = 50_000


def bench(func):
    t0 = time.time()
    for _ in range(N):
        func(S0, K, r, sigma, T)
    return time.time() - t0


print("BS time:", bench(price_call_bs))
print("CRR time:", bench(price_call_crr))
