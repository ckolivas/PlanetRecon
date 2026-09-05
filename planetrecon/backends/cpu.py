"""NumPy/SciPy float64 reference backend."""

from __future__ import annotations

from planetrecon.backends.base import Backend
from planetrecon.runtime import apply_thread_limits


class CPUBackend(Backend):
    name = "cpu"
    precision = "float64"

    def __init__(self, threads: int | None = None):
        self.threads = apply_thread_limits(threads)
