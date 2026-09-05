"""CPU thread limits and device policy.

The numerical reference is CPU float64. A GPU must not be selected for
scientific tests on this development machine. Thread caps are applied through
environment variables before BLAS/FFT pools start; ``threadpoolctl`` is used
when present.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_CPU_THREADS = 8
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def default_thread_count() -> int:
    raw = os.environ.get("PLANETRECON_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return DEFAULT_CPU_THREADS
    return DEFAULT_CPU_THREADS


def apply_thread_limits(n: int | None = None) -> int:
    """Cap BLAS/OpenMP/FFT worker threads. Returns the applied limit."""
    n = default_thread_count() if n is None else max(1, int(n))
    value = str(n)
    os.environ["PLANETRECON_THREADS"] = value
    for key in THREAD_ENV_KEYS:
        os.environ[key] = value
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=n)
    except Exception:
        pass
    return n


def thread_env(n: int | None = None) -> dict[str, str]:
    n = default_thread_count() if n is None else max(1, int(n))
    value = str(n)
    return {key: value for key in THREAD_ENV_KEYS}


def merge_thread_env(env: Mapping[str, str] | None = None, n: int | None = None) -> dict[str, str]:
    out = dict(os.environ if env is None else env)
    out.update(thread_env(n))
    out["PLANETRECON_THREADS"] = str(
        default_thread_count() if n is None else max(1, int(n))
    )
    out["CUDA_VISIBLE_DEVICES"] = ""
    out["HIP_VISIBLE_DEVICES"] = ""
    return out
