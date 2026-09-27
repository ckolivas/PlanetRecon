"""Test native-size mono square-drop deposition at fixed global translations.

This is a diagnostic implementation of explicitly specified kernels, not an
implementation claim about AutoStakkert. Displacement uses PR's output-to-input
convention; an input pixel therefore deposits at input minus displacement.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from planetrecon.export import ExportConfig, export_result
from planetrecon.io.ser import SERSource
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot
from tools.alignment_noise_experiment import read_png
from tools.coherent_saturn_experiment import inputs


def deposit(raw, shift_xy, pixfrac):
    """Area-overlap deposition of square drops at 1x; zero outside detector.

    pixfrac zero is point deposition to the nearest output pixel. Positive
    fractions up to one conserve each interior input pixel's total weight.
    Coverage division occurs after stacking, never as a frame brightness fit.
    """
    if not 0 <= pixfrac <= 1:
        raise ValueError('pixel fraction must lie between zero and one')
    h, w = raw.shape
    total, support = np.zeros((h, w)), np.zeros((h, w))

    def axis(shift):
        centre = -float(shift)
        if pixfrac == 0:
            return [(int(np.rint(centre)), 1.)]
        lo = int(np.floor(centre))
        return [(d, max(0., min(centre+pixfrac/2, d+.5)-max(centre-pixfrac/2, d-.5))/pixfrac)
                for d in (lo, lo+1)]

    for dy, wy in axis(shift_xy[1]):
        for dx, wx in axis(shift_xy[0]):
            weight = wx*wy
            if weight == 0:
                continue
            sy0, sy1 = max(0, -dy), min(h, h-dy)
            sx0, sx1 = max(0, -dx), min(w, w-dx)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            dest = np.s_[sy0+dy:sy1+dy, sx0+dx:sx1+dx]
            total[dest] += weight*raw[sy0:sy1, sx0:sx1]
            support[dest] += weight
    return total, support


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root, indices, quality, shifts, reference, hashes = inputs()
    hashes['deposit_code'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    policies = dict(square_1=1., square_half=.5, point=0.)
    totals = {name: np.zeros_like(reference) for name in policies}
    supports = {name: np.zeros_like(reference) for name in policies}
    # Check the operator independently against SciPy before a real replay.
    rng = np.random.default_rng(9022)
    test = rng.normal(size=(53, 61))
    yy, xx = np.indices(test.shape, dtype=float)
    parity = []
    for shift in [(0., 0.), (2., -1.), (.25, .5), (-.7, .35)]:
        total, weight = deposit(test, shift, 1.)
        coords = np.array([yy+shift[1], xx+shift[0]])
        expected = map_coordinates(test, coords, order=1, mode='grid-constant', cval=0., prefilter=False)
        coverage = map_coordinates(np.ones_like(test), coords, order=1, mode='grid-constant', cval=0., prefilter=False)
        np.testing.assert_allclose(total, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(weight, coverage, atol=1e-12, rtol=0)
        parity.append(float(np.max(abs(total-expected))))
    with SERSource('2024-09-27-1154_3-CK-R-Sat.ser') as source:
        for i, (index, shift, q) in enumerate(zip(indices, shifts, quality)):
            raw = source.read_raw(int(index)).astype(float)
            for name, fraction in policies.items():
                total, weight = deposit(raw, shift, fraction)
                totals[name] += max(float(q), 1e-12)*total
                supports[name] += max(float(q), 1e-12)*weight
            if (i+1) % 512 == 0:
                print(i+1, 'frames deposited', flush=True)
    previous = load_snapshot(root/'global_subpixel.npz')
    _, text = read_png(root/'local.png')
    mapping = json.loads(text)['mapping']
    export = ExportConfig('png16', mapping['black'], mapping['white'], 1.)
    report = dict(n_used=len(indices), hashes=hashes, config=dict(scale=1., pixel_fractions=policies),
                  scipy_parity_max_adu=max(parity), normalization=False, output_filtering=False,
                  production_changed=False, variants={})
    for name in policies:
        support = supports[name]
        pixels = np.divide(totals[name], support, out=np.zeros_like(support), where=support>0)
        result = ReconstructionResult(image=pixels, coverage=support, validity=support>0,
            units=previous.units, channel_order='mono', backend='cpu', precision='float64',
            stage='diagnostic', incomplete=False, reference_epoch=previous.reference_epoch, n_used=len(indices),
            provenance=dict(diagnostic_only=True, kernel=name, config=report['config'], hashes=hashes))
        save_snapshot(args.out/f'{name}.npz', result)
        export_result(result, args.out/f'{name}.png', export)
        values = {}
        if name in ('square_1', 'point'):
            baseline = previous if name == 'square_1' else load_snapshot(root/'global_integer.npz')
            values['previous_stack_max_difference_adu'] = float(np.max(abs(pixels-baseline.image)))
            np.testing.assert_allclose(pixels, baseline.image, atol=1e-10, rtol=0)
        report['variants'][name] = values
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Native unit-drop and point-deposition stacks reproduce the previous bilinear and integer controls.')


if __name__ == '__main__':
    main()
