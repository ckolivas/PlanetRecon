"""Freeze observed-data rankings and all full-development selection fractions."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from planetrecon.rank import score_sequence, subsets_from_scores
from tools.input_compatibility import archived_input_errors
from tools.study_io import identities, open_study, write_json, file_hash

FRACTIONS = (5, 10, 25, 50, 100)


def selection_rows(images, read_noise, strength=.0003):
    images = np.asarray(images, dtype=float)
    if images.ndim != 3 or len(images) != 500 or min(images.shape[1:]) <= 4 or not np.isfinite(images).all():
        raise ValueError('500 finite detector frames with valid interior required')
    if not np.isfinite(read_noise) or read_noise < 0 or not np.isfinite(strength) or strength <= 0:
        raise ValueError('invalid read noise or prior strength')
    scores = score_sequence(images, border=2)
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite ranking scores')
    variance = np.maximum(images.mean(axis=(1, 2)), 0.) + read_noise**2
    if np.any(variance <= 0):
        raise ValueError('positive observed scalar variances required')
    rows = []
    for fraction, indices in subsets_from_scores(scores, FRACTIONS).items():
        rows.append({'fraction': fraction, 'indices': indices.tolist(), 'n_used': len(indices),
                     'observed_electron_sum': float(images[indices].sum()),
                     'mean_observed_scalar_variance': float(variance[indices].mean()),
                     'mean_ridge': strength/len(indices)})
    return scores.tolist(), rows


def read_case(path, crop):
    # Intentionally do not load latent/expected pixels, PSFs, registration or masks.
    with h5py.File(path) as f:
        images = f[f'frames/{crop}_observed_e'][...]
        noise = float(f['config'].attrs['read_noise_e'])
    return selection_rows(images, noise)


def run(inputs, directory):
    from planetrecon.hdf5io import filename
    deps = [Path(__file__), Path('docs/scene-selection-manifest-protocol.md')]
    identity = identities(deps)
    paths = [(s, d, inputs/filename(s, d)) for s in (1001, 1002, 1003) for d in (4., 8.)]
    hashes = {p.name: file_hash(p) for _, _, p in paths}
    protocol = {'identities': identity, 'input_sha256': hashes, 'fractions': list(FRACTIONS),
                'ranking': 'raw observed mean squared Laplacian; border2; later-index ties first; no registration',
                'sum_native_strength': .0003, 'q3_authorized': False,
                'scope': 'Selection manifest only; no reconstruction, truth, held-out assessment or scientific pass.'}
    directory = open_study(directory, protocol)
    rows = []
    for seed, dr0, path in paths:
        if errors := archived_input_errors(path):
            raise ValueError(errors)
        for crop in ('feature', 'bland'):
            scores, selections = read_case(path, crop)
            row = {'seed': seed, 'dr0': dr0, 'crop': crop, 'n_captured': 500,
                   'input': path.name, 'input_sha256': hashes[path.name],
                   'scores': scores, 'selections': selections}
            rows.append(row)
            write_json(directory/f'{seed}-{int(dr0)}-{crop}.json', row)
    unchanged = identity == identities(deps) and all(file_hash(p) == hashes[p.name] for _, _, p in paths)
    report = {'status': 'valid' if unchanged else 'incomplete', 'cases': rows,
              'source_input_unchanged': unchanged, 'q3_authorized': False, 'scope': protocol['scope']}
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.exit(0 if run(args.inputs, args.out)['status'] == 'valid' else 1)
