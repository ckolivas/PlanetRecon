"""Deterministic RNG streams.

Screen draws are paired across seeing regimes: the same seed produces the
same unit Kolmogorov field, which is then scaled to the requested r0.

Noise draws are independent between regimes and are functions of
(seed, regime, frame_index).
"""

from __future__ import annotations

import numpy as np

from planetrecon.config import SimConfig


def screen_rng(seed: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(seed), 0x5C8EE11])
    return np.random.default_rng(ss)


def noise_rng(cfg: SimConfig) -> np.random.Generator:
    dr0_code = int(round(cfg.dr0 * 1000.0))
    ss = np.random.SeedSequence([int(cfg.seed), dr0_code, 0xA11E])
    return np.random.default_rng(ss)


def frame_noise(rng: np.random.Generator, expected: np.ndarray, read_rms: float):
    lam = np.clip(expected, 0.0, None)
    shot = rng.poisson(lam).astype(np.float64)
    read = rng.normal(0.0, read_rms, size=expected.shape)
    return shot + read
