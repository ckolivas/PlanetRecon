"""Scoped CUDA caching-allocator limits; driver/FFT-library memory is excluded."""
from contextlib import contextmanager
import threading

_LOCK = threading.RLock()


@contextmanager
def cuda_allocation_limit(max_bytes):
    # Torch's setting is process-wide. Serialize PlanetRecon GPU stacks within
    # a process; GUI jobs already run in separate owned processes.
    with _LOCK:
        report = {'requested_bytes': max_bytes, 'enforced': False,
                  'scope': 'PyTorch CUDA caching allocator; excludes driver and external library allocations',
                  'effective_bytes': None, 'peak_reserved_bytes': None, 'error': None}
        previous = None
        torch = None
        try:
            if max_bytes is not None:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError('CUDA unavailable')
                    if torch.cuda.memory.get_allocator_backend() != 'native':
                        raise RuntimeError('CUDA allocation budgets require the native PyTorch allocator')
                    total = torch.cuda.get_device_properties(0).total_memory
                    previous = torch.cuda.get_per_process_memory_fraction(0)
                    fraction = min(previous, float(max_bytes / total), 1.)
                    torch.cuda.empty_cache()
                    if torch.cuda.memory_reserved(0) > int(total * fraction):
                        raise RuntimeError('existing live CUDA allocations exceed the requested budget')
                    torch.cuda.set_per_process_memory_fraction(fraction, 0)
                    torch.cuda.reset_peak_memory_stats(0)
                    report.update(enforced=True, effective_bytes=int(total * fraction))
                except Exception as exc:
                    report['error'] = f'CUDA allocation limit unavailable; using CPU: {type(exc).__name__}: {exc}'
            yield report
        finally:
            if previous is not None:
                try:
                    if report['enforced']:
                        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
                finally:
                    torch.cuda.set_per_process_memory_fraction(previous, 0)
                    torch.cuda.empty_cache()
