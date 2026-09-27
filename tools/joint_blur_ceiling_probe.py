"""Sensitivity to the blur ceiling in a few real captured frames.

Uses the frozen fitter. A wider physical blur bank maps its variances into the
existing [0,4] optimization coordinate. Parameter scaling therefore changes
with range; a longer-budget comparison is included and no convergence is
assumed. This is model adequacy evidence, not known geometric truth.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from planetrecon.io.ser import SERSource
from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration
from tools.coherent_saturn_experiment import inputs
from tools.joint_saturn_experiment import digest
from tools.spline_joint_registration import SplineJointRegistration, spline_coefficients


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    _, indices, _, shifts, reference, hashes = inputs()
    torch.set_num_threads(4)
    # Predetermined positions span the fixed ordered selection; do not select
    # cases after seeing which extension gives the most favourable result.
    positions = np.linspace(0, len(indices)-1, 6, dtype=int)
    engines = {}
    for maximum in (2., 4.):
        engine = SplineJointRegistration(reference, CircularMultiscaleRegistration(reference), maxiter=300)
        if maximum != 2.:
            physical = np.arange(0., maximum+.01, .25)
            bank = np.stack([gaussian_filter(reference, sigma) if sigma else reference for sigma in physical])
            engine.bank = torch.as_tensor(spline_coefficients(bank), device=engine.device)
            engine.variances = (physical/(maximum/2))**2
        engines[maximum] = engine
    report = dict(hashes=hashes, source_sha256=digest(__file__), frames=[],
        model_sha256=digest('tools/joint_registration.py'),
        sampler_sha256=digest('tools/spline_joint_registration.py'),
        positions=positions.tolist(), maximum_sigmas=[2., 4.], iteration_budgets=[100, 300])
    with SERSource('2024-09-27-1154_3-CK-R-Sat.ser') as source:
        for pos in positions:
            frame = source.read_raw(int(indices[pos]))
            row = dict(position=int(pos), frame_index=int(indices[pos]), variants=[])
            for maximum, engine in engines.items():
                for budget in (100, 300):
                    engine.maxiter = budget
                    field, stats = engine.fit(frame, shifts[pos], .3, lambda: None)
                    origin = torch.as_tensor(shifts[pos], device=engine.device)[:, None, None]
                    stats['sigma_physical'] = stats['sigma']*(maximum/2)
                    stats['initial_sigma_physical'] = stats['initial_sigma']*(maximum/2)
                    row['variants'].append(dict(maximum_sigma=maximum, budget=budget, stats=stats,
                        residual_vector_rms_px=float((field-origin)[:, engine.mask.bool()].square().sum(0).mean().sqrt())))
            report['frames'].append(row)
            args.out.write_text(json.dumps(report, indent=2)+'\n')
            print(int(pos), [(v['maximum_sigma'], v['budget'], round(v['stats']['sigma_physical'], 3),
                              round(v['stats']['heldout_loss'], 3)) for v in row['variants']], flush=True)


if __name__ == '__main__':
    main()
