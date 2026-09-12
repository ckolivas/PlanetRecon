#!/usr/bin/env python3
"""Profile this process's CUDA Bayer backprojection work, not GPU utilization.

Run serially against two source trees in alternating order. Timed iterations
exclude warmup and profiler overhead. CUDA activities are collected in a separate
two-call pass: their counts describe this instance's work, while their durations
can still change with GPU contention. Summed activity time is not elapsed time.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Result NPZ and adjacent timing JSON')
    parser.add_argument('--source-tree', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_tree.resolve()))
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(4)
    import numpy as np
    import torch
    from planetrecon.backends.torch_accel import TorchBackend
    y, x = np.indices((320, 512), dtype=float)
    shift = (1.3*np.sin(y/47)+.375, .9*np.cos(x/39)-.625)
    raw = np.random.default_rng(213).uniform(0, 255, y.shape)
    backend = TorchBackend()
    for _ in range(3):
        backend.backproject(raw, shift, 'RGGB')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cpu = time.process_time_ns()
    wall = time.perf_counter_ns()
    for _ in range(40):
        output = backend.backproject(raw, shift, 'RGGB')
    torch.cuda.synchronize()
    elapsed = (time.perf_counter_ns()-wall)/1e9
    cpu_seconds = (time.process_time_ns()-cpu)/1e9
    report = dict(pid=os.getpid(), source_tree=str(args.source_tree.resolve()),
                  device=torch.cuda.get_device_name(), iterations=40,
                  process_cpu_seconds=cpu_seconds, elapsed_seconds=elapsed,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(2):
            backend.backproject(raw, shift, 'RGGB')
        torch.cuda.synchronize()
    events = [e for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not events:
        raise RuntimeError('CUDA activity tracing unavailable; no GPU timing claim can be made')
    report.update(profiled_iterations=2, cuda_activity_count=len(events),
                  summed_cuda_activity_us=sum(e.time_range.elapsed_us() for e in events),
                  cuda_activity_names=dict(Counter(e.name for e in events)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez(args.output, **{str(i): value for i, value in enumerate(output)})
    print(json.dumps({k: v for k, v in report.items() if k != 'cuda_activity_names'}), flush=True)


if __name__ == '__main__':
    main()
