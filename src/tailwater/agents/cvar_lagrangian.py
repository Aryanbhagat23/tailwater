"""CVaR constraint via a Lagrange multiplier.

Constraint: CVaR_alpha(Y) >= floor, where CVaR_alpha is the mean of the
lowest alpha-fraction of season yields. Using the Rockafellar-Uryasev form

    CVaR_alpha(Y) = max_nu  nu - E[(nu - Y)^+] / alpha,

the Lagrangian of  max E[R]  s.t.  CVaR >= floor  adds to each season's return

    (lambda / alpha) * min(Y - nu, 0)

with nu the batch's alpha-quantile (VaR) of Y — only seasons in the tail
receive the extra yield incentive, scaled by 1/alpha. The multiplier is
updated by projected dual ascent on the batch's empirical CVaR:

    lambda <- clip(lambda + lr * (floor - CVaR_batch), 0, lambda_max)

so it rises automatically while the tail violates the floor and decays once
the floor is met (Chow, Ghavamzadeh, Janson & Pavone 2017).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CVaRLagrangian:
    floor: float | None            # None = unconstrained (risk-neutral) agent
    alpha: float = 0.05
    lr: float = 2.0
    lam: float = 0.0
    lam_max: float = 20.0

    def tail(self, yields):
        """(tail mask, VaR nu, empirical CVaR) of a batch of season yields."""
        y = np.asarray(yields)
        k = max(1, math.ceil(self.alpha * len(y)))
        order = np.sort(y)
        nu = order[k - 1]
        return y <= nu, float(nu), float(order[:k].mean())

    def episode_weights(self, yields):
        """Per-season extra weight on yield loss: lambda/alpha for tail seasons."""
        mask, nu, cvar = self.tail(yields)
        if self.floor is None:
            return np.zeros(len(mask)), nu, cvar, mask
        return (self.lam / self.alpha) * mask, nu, cvar, mask

    def update(self, cvar_batch: float):
        if self.floor is None:
            return self.lam
        self.lam = float(np.clip(self.lam + self.lr * (self.floor - cvar_batch), 0.0, self.lam_max))
        return self.lam
