"""Poisson pmf and the Dixon-Coles low-score correction."""

from __future__ import annotations

import math


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    if k < 0:
        return 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles dependence correction for the four low-score cells.

    Independent Poissons under-predict 0-0 and 1-1 and mis-handle 1-0/0-1;
    tau re-weights exactly those cells (and is 1 everywhere else).
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0
