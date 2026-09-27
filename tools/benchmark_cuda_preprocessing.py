"""Read-only CPU/CUDA preprocessing replay; existing capture caches are untouched.

Run with python -m tools.benchmark_cuda_preprocessing --path capture.ser.
--full includes hashes, geometry and temporary cache writes for the entire SER;
the default compares warmed screening in CPU/CUDA/CUDA/CPU order on a prefix.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from planetrecon.io.ser import SERSource
from planetrecon.io.source import FrameSource
from planetrecon.pipeline.preprocess import best_frame_mask, screen_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.reconstruction import ReconstructionConfig


class PrefixSource(FrameSource):
    def __init__(self, source, count):
        self.source = source
        self.count = min(count, source.n_frames())

    def n_frames(self):
        return self.count

    def frame_shape(self):
        return self.source.frame_shape()

    def color_mode(self):
        return self.source.color_mode()

    def metadata(self):
        return replace(self.source.metadata(), n_frames=self.count)

    def read_raw(self, index):
        return self.source.read_raw(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--batch-frames', type=int, default=32)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    config = ReconstructionConfig(device='cpu', threads=args.threads, batch_frames=args.batch_frames)
    timings = {'cpu': [], 'gpu': []}
    outputs = {}
    with SERSource(args.path) as original, TemporaryDirectory(prefix='pr-preprocess-bench-') as temporary:
        source = original if args.full else PrefixSource(original, args.frames)
        if not args.full:
            for device in ('cpu', 'gpu'):
                screen_source(PrefixSource(original, 32), replace(config, device=device))
        for run, device in enumerate(('cpu', 'gpu') if args.full else ('cpu', 'gpu', 'gpu', 'cpu')):
            started = perf_counter()
            cfg = replace(config, device=device)
            if args.full:
                result = preprocess_source(source, cfg, cache_path=Path(temporary)/f'{run}.npz')
            else:
                result = screen_source(source, cfg)
            elapsed = perf_counter()-started
            if device == 'gpu' and result.summary['execution']['cuda_frames'] != source.n_frames():
                raise RuntimeError(f'CUDA replay fell back: {result.summary["execution"]}')
            timings[device].append(elapsed)
            outputs[device] = result
            print(json.dumps({'device': device, 'seconds': elapsed}), flush=True)
        cpu, gpu = outputs['cpu'], outputs['gpu']
        differences = np.abs(cpu.measurements-gpu.measurements)
        report = {
            'capture': str(args.path), 'frames': source.n_frames(), 'shape': source.frame_shape(),
            'color': source.color_mode(), 'threads': args.threads, 'batch_frames': args.batch_frames,
            'scope': ('Single CPU then CUDA full preprocessing, including hashes, geometry and temporary cache writes'
                      if args.full else 'Warmed screening including source reads, CPU/CUDA/CUDA/CPU order'),
            'seconds': timings,
            'maximum_metric_differences': [float(column[np.isfinite(column)].max())
                                           if np.isfinite(column).any() else None for column in differences.T],
            'nonfinite_pattern_identical': bool(np.array_equal(np.isnan(cpu.measurements), np.isnan(gpu.measurements))),
            'screening_mask_differences': int(np.count_nonzero(cpu.accepted != gpu.accepted)),
            'rejection_reasons_identical': cpu.summary['rejected_indices_by_reason'] == gpu.summary['rejected_indices_by_reason'],
            'best_reference_identical': cpu.best_reference_index == gpu.best_reference_index,
            'selection_differences': {
                mode: {str(percent): int(np.count_nonzero(best_frame_mask(cpu, percent, mode)
                                                         != best_frame_mask(gpu, percent, mode)))
                       for percent in range(1, 101)} for mode in ('frame_count', 'quality_range')},
        }
        if args.full:
            report['geometry_identical'] = cpu.summary['geometry_estimate'] == gpu.summary['geometry_estimate']
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        print(f'Report: {args.out}')


if __name__ == '__main__':
    main()
