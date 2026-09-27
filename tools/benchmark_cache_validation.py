"""Read-only full-capture validation timings; never edit existing caches.

Report wall time, process CPU time and Linux physical read bytes for raw reads,
legacy verification and the parallel identity at each requested CPU count.
No filesystem cache flush is performed: the I/O counters identify warm runs.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

import numpy as np

from planetrecon.io import open_source
from planetrecon.pipeline.capture_hash import LEGACY_HASH_METHOD
from planetrecon.pipeline.preprocess_cache import identity, load_cache
from planetrecon.reconstruction import ReconstructionConfig


def io_counts():
    try:
        return {key: int(value) for key, value in
                (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    except OSError:
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, help='Also time an upgrade and reloads of a temporary cache copy')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    cfg = ReconstructionConfig(device='cpu')
    records, reference = [], None
    with open_source(args.path) as source:
        def measure(mode, threads):
            nonlocal reference
            counts, cpu, start = io_counts(), time.process_time(), time.perf_counter()
            if mode == 'read_raw':
                for i in range(source.n_frames()):
                    source.read_raw(i)
            else:
                options = {'hash_method': LEGACY_HASH_METHOD} if mode == 'legacy_identity' else {}
                result = identity(source, replace(cfg, threads=threads), None, **options)
                if not options:
                    if reference is None:
                        reference = result
                    assert result == reference, 'identity changed with CPU count'
            elapsed, used_cpu, after = time.perf_counter()-start, time.process_time()-cpu, io_counts()
            record = {'mode': mode, 'threads': threads, 'wall_s': elapsed, 'cpu_s': used_cpu,
                      'physical_read_bytes': (after['read_bytes']-counts['read_bytes']
                                              if 'read_bytes' in counts else None)}
            records.append(record)
            print(json.dumps(record), flush=True)
        for iteration in range(args.repeats):
            measure('legacy_identity', 1)
            measure('read_raw', 1)
            for threads in args.threads if iteration % 2 == 0 else reversed(args.threads):
                measure('parallel_identity', threads)
        report = {'capture': str(args.path), 'n_frames': source.n_frames(),
                  'shape': source.frame_shape(), 'file_bytes': args.path.stat().st_size,
                  'scope': 'Full observed-pixel identity including timestamps and metadata; no cache flush',
                  'identities_equal_across_worker_counts': True, 'runs': records}
    if args.cache:
        with TemporaryDirectory(prefix='pr-cache-bench-') as directory:
            copied = Path(directory)/'cache.npz'
            shutil.copyfile(args.cache, copied)
            with np.load(copied) as data:
                original = json.loads(str(data['metadata']))
            reloads = []
            for mode in ('legacy_upgrade', 'fast_new_session', 'force_full', 'fast_new_session'):
                with open_source(args.path) as source:
                    start, cpu = time.perf_counter(), time.process_time()
                    selected, status = load_cache(source, replace(cfg, threads=max(args.threads)), path=copied,
                                                  force_full_validation=mode == 'force_full')
                    record = {'mode': mode, 'wall_s': time.perf_counter()-start,
                              'cpu_s': time.process_time()-cpu, 'status': status['status']}
                    assert selected is not None, status
                    assert selected.digest == original['digest'] and selected.identity == original['identity']
                    reloads.append(record)
                    print(json.dumps(record), flush=True)
            report['existing_cache_roundtrip'] = {
                'scope': 'Temporary cache copy; original unchanged; no session receipts',
                'selection_digest_preserved': True, 'runs': reloads}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
