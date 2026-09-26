"""Build disjoint, time-interleaved Saturn half stacks with separate references.

Reuses the exact selected IDs from alignment_noise_experiment.py. This is a
diagnostic run; it never changes the capture's preprocessing cache.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from planetrecon.export import ExportConfig, export_result
from planetrecon.io.ser import SERSource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import load_cache, selection_digest
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import save_snapshot
from planetrecon.runtime import apply_thread_limits
from alignment_noise_experiment import read_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    apply_thread_limits(8)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment')
    torch.set_num_threads(8)
    prior = json.loads((args.experiment/'report.json').read_text())
    config = replace(ReconstructionConfig.from_dict(prior['config']),
                     device='gpu', threads=8, reference_index=0, stack_percent=100)
    with np.load(args.experiment/'diagnostics.npz') as data:
        indices = data['indices'].copy()
    if len(indices) % 2:
        raise ValueError('This controlled run requires an even selected frame count')
    # Randomise membership within adjacent selected-frame pairs, preserving
    # time coverage without putting every earlier member into the same half.
    rng = np.random.default_rng(260926)
    pairs = indices.reshape(-1, 2)
    choice = rng.integers(0, 2, len(pairs))
    halves = [np.sort(pairs[np.arange(len(pairs)), choice]),
              np.sort(pairs[np.arange(len(pairs)), 1-choice])]
    assert not np.intersect1d(*halves).size
    np.testing.assert_array_equal(np.sort(np.concatenate(halves)), indices)
    _, metadata = read_png(args.experiment/'local.png')
    mapping = json.loads(metadata)['mapping']
    report = dict(seed=260926, source_experiment=str(args.experiment),
                  split='one random member of each consecutive selected-frame pair',
                  reference='separate best 64 from each disjoint half', halves=[])
    started = time.perf_counter()
    with SERSource(args.capture) as source:
        selection, status = load_cache(source, config)
        if selection is None:
            raise ValueError(status)
        for half, ids in enumerate(halves):
            accepted = np.zeros_like(selection.accepted)
            accepted[ids] = True
            assert np.all(selection.accepted[ids])
            custom = replace(selection, accepted=accepted,
                summary={**selection.summary, 'diagnostic_subset': dict(
                    half=half, count=len(ids), seed=260926,
                    note='Original screening summary retained; accepted mask restricted for experiment')})
            custom.digest = selection_digest(custom)
            last = [-1]

            def progress(result, info):
                bucket = result.n_used // 512
                if bucket != last[0]:
                    print(json.dumps(dict(half=half, used=result.n_used,
                        seconds=round(time.perf_counter()-started, 1))), flush=True)
                    last[0] = bucket

            result = stack_source(source, config, preprocessing=custom, on_event=progress)
            assert result.backend == 'cuda' and result.n_used == len(ids) and not result.incomplete
            candidates = result.provenance['local_alignment']['template_candidates']
            assert len(candidates) == 64 and np.all(np.isin(candidates, ids))
            result.provenance['independent_half_experiment'] = dict(
                half=half, seed=260926, selected_ids_file=f'half{half}_indices.npy')
            save_snapshot(args.out/f'half{half}.npz', result)
            np.save(args.out/f'half{half}_indices.npy', ids)
            export_result(result, args.out/f'half{half}.png',
                          ExportConfig('png16', mapping['black'], mapping['white']))
            report['halves'].append(dict(n_used=result.n_used, candidates=candidates,
                anchor=result.provenance['local_alignment']['anchor_index'],
                config=config.to_dict(), backend=result.backend))
    assert not set(report['halves'][0]['candidates']) & set(report['halves'][1]['candidates'])
    report['seconds'] = time.perf_counter()-started
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(complete=True, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
