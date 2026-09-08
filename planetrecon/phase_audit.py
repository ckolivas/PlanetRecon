"""Separate phase-basis truncation from finite-exposure error on physical screens.

This bounded development diagnostic never authorizes Q3 or changes the frozen
simulator. All comparisons use the same detector integration, crop and flux.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from planetrecon import constants as C
from planetrecon.atmosphere import extract_phase, finite_exposure_psf, generate_screen
from planetrecon.config import make_config
from planetrecon.mfbd import PupilForward
from planetrecon.optics import bin_box, center_crop, instantaneous_psf, make_pupil, otf_from_centered_psf
from planetrecon.provenance import source_hash
from planetrecon.rng import screen_rng
from planetrecon.runtime import apply_thread_limits


def detector_otf(fwd, optical_psf):
    image = center_crop(bin_box(optical_psf, fwd.bin_factor), fwd.eval_size)
    return otf_from_centered_psf(image/image.sum())


def relative_error(model, reference):
    return float(np.linalg.norm(model-reference)/np.linalg.norm(reference))


def decompose(fwd, midpoint_phase, exposure_psf, refined_exposure_psf, modes=(15, 35, 60)):
    exact_mid = detector_otf(fwd, instantaneous_psf(fwd.amplitude, midpoint_phase))
    exposure = detector_otf(fwd, exposure_psf)
    refined = detector_otf(fwd, refined_exposure_psf)
    phase = midpoint_phase[fwd.mask]
    phase = phase-phase.mean()
    rows = []
    for count in modes:
        basis = fwd.modes[:, :count]
        coeff = basis.T @ phase
        projected = fwd.otf(coeff)
        rows.append({'modes': count, 'basis_only_relative_otf_error': relative_error(projected, exact_mid),
                     'combined_relative_otf_error': relative_error(projected, exposure),
                     'phase_residual_rms_rad': float(np.sqrt(np.mean((phase-basis@coeff)**2)))})
    return {'exposure_only_relative_otf_error': relative_error(exact_mid, exposure),
            'exposure_quadrature_relative_otf_error': relative_error(exposure, refined), 'basis': rows}


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    apply_thread_limits(2)
    protocol = {'status': 'diagnostic', 'q3_authorized': False, 'seeds': list(C.DEV_SEEDS),
                'regimes': [4., 8.], 'n_frames': 8, 'n_diam': 16, 'pupil_pad_factor': 8.,
                'eval_size': 32, 'exposure_samples': [8, 16], 'modes': [15, 35, 60],
                'metric': 'relative L2 norm over the full complex detector OTF',
                'design': 'Same physical screen and detector operator for exact midpoint, exposure average and projected midpoint; errors are not additive.',
                'limits': 'Reduced pupil and detector grids; no object optimization, ranking, independent assessment or Gate qualification.'}
    (directory/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    cases = []
    for seed in protocol['seeds']:
        for regime in protocol['regimes']:
            cfg = make_config(seed, regime, n_frames=8, n_diam=16, pupil_pad_factor=8., eval_size=32)
            pupil = make_pupil(cfg)
            fwd = PupilForward.from_config(cfg)
            screen = generate_screen(cfg, screen_rng(seed))
            rows = []
            for index in range(cfg.n_frames):
                t0 = index*cfg.dt_s
                phase = extract_phase(pupil, screen, t0+.5*cfg.texp_s, cfg.wind_m_s)
                exposure = finite_exposure_psf(pupil, screen, t0, cfg.texp_s, 8, cfg.wind_m_s)
                refined = finite_exposure_psf(pupil, screen, t0, cfg.texp_s, 16, cfg.wind_m_s)
                rows.append({'frame_index': index, **decompose(fwd, phase, exposure, refined)})
            cases.append({'config': asdict(cfg), 'frames': rows})
    report = {'protocol': protocol, 'source_hash': source_hash(), 'cases': cases,
              'wall_time_s': time.monotonic()-started}
    (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    report = run(parser.parse_args().out)
    print(json.dumps({'cases': len(report['cases']), 'wall_time_s': report['wall_time_s']}, indent=2))
