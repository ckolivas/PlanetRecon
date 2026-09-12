#!/usr/bin/env python3
"""Time one stacking replay, attributing CPU work to this process only.

Input NPZ: frames (8-bit RGGB), times (seconds), measurements (cached
preprocessing rows), and config (a JSON ReconstructionConfig). Use an already
selected frame set. Replays use local alignment with 65-pixel patches and retain
all sampled frames; the stored geometry, rates and exposure remain in effect.
Each invocation runs in its own process. Compare unchanged inputs/configuration
in alternating before/after order; do not run benchmark instances concurrently.
Input loading, imports, CUDA warmup and result export are outside the timer.
Wall time includes contention; process CPU time includes this process's threads
only. Neither is a measurement of exclusive GPU execution time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import replace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path, help='Result NPZ; timing JSON is saved alongside it')
    parser.add_argument('--source-tree', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mode', choices=['none', 'surface', 'combined', 'saturn'], required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--threads', type=int, default=32, choices=range(1, 33))
    parser.add_argument('--samples', type=int, default=33, help='Uniform samples, plus the best frame')
    parser.add_argument('--synthetic-flat', action='store_true',
                        help='Exercise calibration with a generated positive flat; not a measured capture calibration')
    args = parser.parse_args()
    if args.samples < 4:
        parser.error('--samples must be at least 4')
    sys.path.insert(0, str(args.source_tree.resolve()))
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(args.threads)
    import numpy as np
    from planetrecon.io.source import ArraySource
    from planetrecon.reconstruction import ReconstructionConfig
    from planetrecon.pipeline.preprocess import FrameSelection
    from planetrecon.pipeline.preprocess_cache import identity, selection_digest
    from planetrecon.pipeline.baseline import stack_source

    with np.load(args.input, allow_pickle=False) as data:
        indices = np.unique(np.r_[np.linspace(0, len(data['frames'])-1, args.samples).astype(int),
                                   np.argmax(data['measurements'][:, 0])])
        source = ArraySource(data['frames'][indices], color_mode='RGGB', bit_depth=8,
                             timestamps=data['times'][indices])
        config = ReconstructionConfig.from_dict(json.loads(str(data['config'])))
        measurements = data['measurements'][indices].copy()
    config = replace(config, frame_preselection=True, local_alignment=True,
                     local_patch_size=65, threads=args.threads, reference_index=0,
                     geometry_mode=args.mode, device=args.device, stack_percent=100,
                     **(dict(ring_inner_radius_px=None, ring_outer_radius_px=None)
                        if args.mode != 'saturn' else {}))
    selection = FrameSelection(np.ones(source.n_frames(), bool), measurements,
        {'rejected_indices_by_reason': {key: [] for key in
         ('low_quality', 'width_outlier', 'height_outlier', 'clipped_target',
          'no_target', 'invalid', 'saturated')}})
    calibration = None
    if args.synthetic_flat:
        from planetrecon.calibration import Calibration
        h, w = source.frame_shape()
        y, x = np.indices((h, w), dtype=float)
        calibration = Calibration(flat=.8+.2*x/max(1,w-1)+.1*y/max(1,h-1))
    selection.identity = identity(source, config, calibration)
    selection.digest = selection_digest(selection)
    torch = None
    if args.device == 'gpu':
        import torch
        # Fail visibly when qualifying CUDA on a machine without a usable GPU.
        torch.ones(1, device='cuda').sum().item()
        torch.cuda.synchronize()
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    result = stack_source(source, config, preprocessing=selection, calibration=calibration)
    if torch is not None:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter_ns()-start)/1e9
    cpu_seconds = (time.process_time_ns()-cpu_start)/1e9
    report = dict(pid=os.getpid(), source_tree=str(args.source_tree.resolve()),
                  synthetic_flat=args.synthetic_flat,
                  input=str(args.input.resolve()), input_indices=indices.tolist(),
                  config=config.to_dict(), backend=result.backend, n_used=result.n_used,
                  process_cpu_seconds=cpu_seconds, elapsed_seconds=elapsed,
                  timing_scope='stack_source including template construction; excludes input load, '
                               'imports, CUDA warmup and output serialization',
                  cpu_attribution='this process and its threads only; no child processes are started',
                  elapsed_caveat='includes scheduling, I/O and GPU contention; not exclusive compute time')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, image=result.image, coverage=result.coverage,
             validity=result.validity, n_used=result.n_used)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
