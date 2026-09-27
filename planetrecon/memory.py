"""Conservative Linux CPU process memory ceiling via RLIMIT_AS.

This includes mapped address space, not just resident pages. Run capped jobs in
owned processes: the OS setting applies to every thread in the current process.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import sys
import threading

_LOCK = threading.RLock()


def total_ram_bytes():
    """Total physical RAM, independent of current usage and filesystem caches.

    Fall back to the physical page count where MemTotal is unavailable. An
    unknown value returns None so callers can avoid speculative parallelism.
    """
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == 'MemTotal:':
                if len(fields) == 3 and fields[2] == 'kB':
                    value = int(fields[1])
                    if value >= 0:
                        return value * 1024
                break
    except (OSError, ValueError):
        pass
    try:
        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        if pages >= 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    return None


def virtual_bytes():
    return int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')


@contextmanager
def cpu_memory_limit(max_bytes):
    with _LOCK:
        if max_bytes is None:
            yield None
            return
        if sys.platform != 'linux':
            raise ValueError('CPU process memory limits are currently qualified on Linux only')
        import resource
        old = resource.getrlimit(resource.RLIMIT_AS)
        effective = min([max_bytes] + [value for value in old if value != resource.RLIM_INFINITY])
        current = virtual_bytes()
        if current >= effective:
            raise MemoryError(f'CPU memory cap {effective} bytes is below the existing mapped address space {current}; increase the cap')
        report = {'requested_bytes': max_bytes, 'effective_bytes': effective, 'enforced': True,
                  'scope': 'Linux process RLIMIT_AS; includes mapped libraries; excludes GUI parent and other processes',
                  'initial_virtual_bytes': current, 'process_peak_rss_bytes': None}
        resource.setrlimit(resource.RLIMIT_AS, (effective, old[1]))
        try:
            yield report
        finally:
            # Restore first so exception reporting can allocate even after ENOMEM.
            resource.setrlimit(resource.RLIMIT_AS, old)
            report['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
