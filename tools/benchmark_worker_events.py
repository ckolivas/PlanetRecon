#!/usr/bin/env python3
"""Measure owned worker CPU/serialization with and without preview subscription.

Uses one process and an immediately consuming transport that serializes each
event with the production pickle protocol. Includes capture reading/stacking and
payload serialization, excludes startup imports, output saving and disk spooling.
The simulated consumer requests a full snapshot at every batch. This isolates
unused preview work; it is not a full GUI latency benchmark. Compare serial runs
in alternating order on the same input. Background load affects elapsed time.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--no-previews', action='store_true')
    parser.add_argument('--batch-frames', type=int, default=32)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(4)
    import numpy as np
    from planetrecon.jobs import _worker_run
    from planetrecon.pipeline import baseline  # Exclude numerical imports from timing.
    from planetrecon.reconstruction import ReconstructionConfig
    cfg = ReconstructionConfig(device='cpu', threads=4, batch_frames=args.batch_frames,
                               frame_preselection=False, local_alignment=False)
    events, sizes = Counter(), Counter()
    completed = None
    request = threading.Event()
    request.set()
    cancelled = threading.Event()
    class Queue:
        def put(self, event, **kwargs):
            nonlocal completed
            payload = pickle.dumps(event, protocol=5)
            events[event.kind] += 1
            sizes[event.kind] += len(payload)
            if event.kind == 'snapshot':
                request.set()
            elif event.kind == 'completed':
                completed = event.payload
            elif event.kind == 'error':
                raise RuntimeError(event.payload['message'])
        put_nowait = put
        def cancel_join_thread(self):
            pass
    cpu, wall = time.process_time_ns(), time.perf_counter_ns()
    _worker_run('event-benchmark', str(args.input), cfg.to_dict(), Queue(), cancelled,
                None, request, emit_previews=not args.no_previews)
    elapsed, cpu_seconds = (time.perf_counter_ns()-wall)/1e9, (time.process_time_ns()-cpu)/1e9
    if completed is None:
        raise RuntimeError('worker did not complete')
    report = dict(pid=os.getpid(), emit_previews=not args.no_previews,
                  batch_frames=args.batch_frames, process_cpu_seconds=cpu_seconds,
                  elapsed_seconds=elapsed, events=dict(events), serialized_bytes=dict(sizes),
                  n_used=completed['n_used'], n_rejected=completed['n_rejected'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{key: completed[key] for key in ('image', 'coverage', 'validity')})
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
