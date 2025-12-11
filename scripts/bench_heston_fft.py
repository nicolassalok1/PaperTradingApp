import time
import numpy as np

from app.model.heston.pricing.heston_fft import heston_fft_price

S0 = 100
K = 100
r = 0.01
v0 = 0.04
rho = -0.7
kappa = 1.5
theta = 0.04
sigma_v = 0.5
T = 1.0

N = 500  # évite de carboniser ton PC


def bench_fft():
    t0 = time.time()
    for _ in range(N):
        heston_fft_price(S0, K, r, v0, rho, kappa, theta, sigma_v, T)
    return time.time() - t0


print("Heston FFT time:", bench_fft())
