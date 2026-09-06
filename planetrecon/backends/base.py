"""Device selection with a live operator probe. CPU is always available."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from planetrecon.runtime import DEFAULT_CPU_THREADS


DeviceChoice = Literal["cpu", "auto", "gpu"]


@dataclass
class DeviceReport:
    requested: str
    selected: str
    available: list[str]
    fallback: bool
    reason: str = ""
    name: str = "cpu"
    warnings: list[str] = field(default_factory=list)


class Backend:
    name = "cpu"
    precision = "float64"

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)

    def phase_correlation(self, reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
        from planetrecon.pipeline.align import phase_correlation_shift

        return phase_correlation_shift(reference, frame)

    def shift(self, image: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
        from scipy.ndimage import shift as ndshift

        sy, sx = float(shift_xy[1]), float(shift_xy[0])
        return ndshift(np.asarray(image, dtype=np.float64), shift=(sy, sx), order=1, prefilter=False)


def probe_torch_cuda() -> DeviceReport:
    name = "cpu"
    try:
        import warnings

        import torch

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")
            cuda_ok = bool(torch.cuda.is_available())
    except Exception as exc:
        return DeviceReport("gpu", "cpu", ["cpu"], True, f"torch_not_importable:{exc}", "cpu")
    if not cuda_ok:
        return DeviceReport("gpu", "cpu", ["cpu"], True, "cuda_not_available", "cpu")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            sm = f"sm_{major}{minor}"
            archs = list(torch.cuda.get_arch_list())
    except Exception as exc:
        return DeviceReport(
            "gpu", "cpu", ["cpu"], True, f"capability_query_failed:{exc}", name, [str(exc)]
        )
    if sm not in archs and f"compute_{major}{minor}" not in archs:
        reason = (
            f"cuda_arch_unsupported:{sm} not in {archs}; "
            f"Torch {getattr(torch, '__version__', '?')} cannot run {name}"
        )
        return DeviceReport("gpu", "cpu", ["cpu"], True, reason, name, [reason])
    try:
        x = torch.randn(32, 32, device="cuda:0", dtype=torch.float64)
        y = torch.fft.fft2(x)
        _ = y.abs().sum().item()
        del x, y
        torch.cuda.synchronize()
    except Exception as exc:
        reason = f"cuda_operator_probe_failed:{type(exc).__name__}"
        return DeviceReport("gpu", "cpu", ["cpu"], True, reason, name, [reason])
    return DeviceReport("gpu", "gpu", ["cpu", "gpu"], False, "ok", name)


def select_backend(choice: DeviceChoice, threads: int = DEFAULT_CPU_THREADS) -> tuple[Backend, DeviceReport]:
    from planetrecon.backends.cpu import CPUBackend

    if choice not in ("cpu", "auto", "gpu"):
        raise ValueError(f"unknown device {choice!r}")
    cpu = CPUBackend(threads=threads)
    if choice == "cpu":
        return cpu, DeviceReport("cpu", "cpu", ["cpu"], False, "explicit_cpu", "cpu")
    report = probe_torch_cuda()
    report.requested = choice
    if report.selected == "gpu":
        from planetrecon.backends.torch_accel import TorchBackend

        return TorchBackend(), report
    if choice == "gpu":
        report.requested = "gpu"
        report.fallback = True
        report.warnings.append(
            "explicit GPU request failed; continue on CPU. "
            + report.reason
        )
        return cpu, report
    report.requested = "auto"
    return cpu, report
