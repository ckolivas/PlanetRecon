"""CPU thread limits and device policy.

The numerical reference is CPU float64. Optional CUDA backends are
selected only after a live operator probe. Thread caps are applied through
environment variables before BLAS/FFT pools start; ``threadpoolctl`` is used
when present.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_CPU_THREADS = 32
MAX_CPU_THREADS = 32
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def detected_thread_count() -> int:
    """Logical CPUs available to this process, within the application ceiling."""
    try:
        detected = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        detected = os.cpu_count() or 1
    return max(1, min(int(detected), MAX_CPU_THREADS))


def default_thread_count() -> int:
    raw = os.environ.get("PLANETRECON_THREADS")
    n = detected_thread_count()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = detected_thread_count()
    n = max(1, min(int(n), MAX_CPU_THREADS))
    return min(n, detected_thread_count())


def apply_thread_limits(n: int | None = None) -> int:
    """Cap BLAS/OpenMP/FFT worker threads. Returns the applied limit."""
    n = default_thread_count() if n is None else max(1, min(int(n), MAX_CPU_THREADS))
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
    n = default_thread_count() if n is None else max(1, min(int(n), MAX_CPU_THREADS))
    value = str(n)
    return {key: value for key in THREAD_ENV_KEYS}


def merge_thread_env(
    env: Mapping[str, str] | None = None,
    n: int | None = None,
    *,
    hide_gpu: bool = False,
) -> dict[str, str]:
    out = dict(os.environ if env is None else env)
    out.update(thread_env(n))
    applied = default_thread_count() if n is None else max(1, min(int(n), MAX_CPU_THREADS))
    out["PLANETRECON_THREADS"] = str(applied)
    if hide_gpu:
        out["CUDA_VISIBLE_DEVICES"] = ""
        out["HIP_VISIBLE_DEVICES"] = ""
    return out
