"""Candidate-only reproducible multiplicative lognormal AR(1) noise."""
from __future__ import annotations

import numpy as np

NOISE_SEED = 20261017
RHO = 0.6
CV = 0.25


def lognormal_ar1_multipliers(basin_ids: np.ndarray, n_days: int, *, seed: int = NOISE_SEED,
                              rho: float = RHO, cv: float = CV) -> np.ndarray:
    """Return stationary mean-one lognormal AR(1) multipliers [basin, day]."""
    basin_ids = np.asarray(basin_ids)
    if basin_ids.ndim != 1 or n_days < 1 or not np.issubdtype(basin_ids.dtype, np.integer):
        raise ValueError("basin_ids must be a 1-D integer array and n_days positive")
    if not (-1.0 < rho < 1.0) or cv < 0:
        raise ValueError("require -1 < rho < 1 and cv >= 0")
    sigma = float(np.sqrt(np.log1p(cv * cv)))
    out = np.empty((len(basin_ids), n_days), dtype=np.float64)
    for row, basin_id in enumerate(basin_ids):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(basin_id)])))
        z = np.empty(n_days, dtype=np.float64)
        z[0] = rng.standard_normal()
        innovation_scale = np.sqrt(1.0 - rho * rho)
        for day in range(1, n_days):
            z[day] = rho * z[day - 1] + innovation_scale * rng.standard_normal()
        out[row] = np.exp(sigma * z - 0.5 * sigma * sigma)
    return out
