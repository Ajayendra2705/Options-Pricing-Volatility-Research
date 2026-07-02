from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Quick FD sanity of Day 3 analytic Greeks before Day 4."""
import sys, os
sys.path.insert(0, ROOT)
import numpy as np
from src.greeks import black_scholes as bs

F, K, T, sig, r = 100.0, 105.0, 0.6, 0.25, 0.04
h = 1e-5

def fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

for cp in (+1, -1):
    name = "call" if cp == 1 else "put"
    d_fd = fd(lambda x: bs.price(x, K, T, sig, r, cp), F, h)
    g_fd = (bs.price(F+h, K, T, sig, r, cp) - 2*bs.price(F, K, T, sig, r, cp) + bs.price(F-h, K, T, sig, r, cp)) / h**2
    v_fd = fd(lambda x: bs.price(F, K, T, x, r, cp), sig, h)
    th_fd = -fd(lambda x: bs.price(F, K, x, sig, r, cp), T, h)   # dV/dt = -dV/dT
    rho_fd = fd(lambda x: bs.price(F, K, T, sig, x, cp), r, h)
    print(f"{name}: delta {bs.delta(F,K,T,sig,r,cp):.8f} vs {d_fd:.8f} | "
          f"gamma {bs.gamma(F,K,T,sig,r):.8f} vs {g_fd:.8f} | "
          f"vega {bs.vega(F,K,T,sig,r):.8f} vs {v_fd:.8f} | "
          f"theta {bs.theta(F,K,T,sig,r,cp):.8f} vs {th_fd:.8f} | "
          f"rho {bs.rho(F,K,T,sig,r,cp):.8f} vs {rho_fd:.8f}")

# spot versions, with carry
S, q = 100.0, 0.015
for cp in (+1, -1):
    name = "call" if cp == 1 else "put"
    d_fd = fd(lambda x: bs.price_spot(x, K, T, sig, r, q, cp), S, h)
    g_fd = (bs.price_spot(S+h, K, T, sig, r, q, cp) - 2*bs.price_spot(S, K, T, sig, r, q, cp) + bs.price_spot(S-h, K, T, sig, r, q, cp)) / h**2
    th_fd = -fd(lambda x: bs.price_spot(S, K, x, sig, r, q, cp), T, h)
    rho_fd = fd(lambda x: bs.price_spot(S, K, T, sig, x, q, cp), r, h)
    print(f"{name}_spot: delta {bs.delta_spot(S,K,T,sig,r,q,cp):.8f} vs {d_fd:.8f} | "
          f"gamma {bs.gamma_spot(S,K,T,sig,r,q):.8f} vs {g_fd:.8f} | "
          f"theta {bs.theta_spot(S,K,T,sig,r,q,cp):.8f} vs {th_fd:.8f} | "
          f"rho {bs.rho_spot(S,K,T,sig,r,q,cp):.8f} vs {rho_fd:.8f}")
