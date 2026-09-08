import numpy as np
import pytest

from planetrecon.config import make_config
from planetrecon.evaluate import CropArrays, evaluate_crop
from planetrecon.optics import otf_from_centered_psf
from tools.audit_full_gate1 import compare, complete_family


def small_crop():
    y, x = np.indices((16, 16))
    truth = 30*np.exp(-((x-7)**2+(y-9)**2)/12.)
    psf = np.exp(-((x-8)**2+(y-8)**2)/3.)
    h = otf_from_centered_psf(psf/psf.sum())
    expected = np.repeat(np.fft.ifft2(h*np.fft.fft2(truth)).real[None], 20, axis=0)
    observed = expected + np.random.default_rng(12).normal(size=expected.shape)
    mask = np.ones_like(truth, dtype=bool)
    mask[0, 0] = False
    crop = CropArrays('bland', truth, observed, expected, np.repeat(h[None], 20, axis=0),
                      np.zeros((20, 2)), np.zeros_like(mask), np.ones_like(truth),
                      np.ones_like(truth), mask, mask, np.ones_like(truth), {}, {})
    extras = {'strehl': np.ones(20), 'phase_rms': np.ones(20), 'read_noise_e': 2.}
    return make_config(1001, 4., n_frames=20, eval_size=16), crop, extras


def test_audit_retains_failed_budget_without_accepting_it():
    args = small_crop()
    with pytest.raises(RuntimeError, match='did not converge'):
        evaluate_crop(*args, maxiter=1)
    result = evaluate_crop(*args, maxiter=1, require_convergence=False, return_reconstructions=True)
    assert result['status'] == 'incomplete'
    assert result['solver_maxiter'] == 1
    assert set(result['_reconstructions']) == {5, 10, 25, 50, 100}
    assert not result['metrics']['100']['_info']['E2a']['converged']


def test_stable_images_cannot_hide_unconverged_or_nonfinite_gaps():
    images = [{10: {'E2a': np.ones((2, 2))}}]*2
    good = {'status': 'valid', **{g: {'g': .2} for g in ('G1', 'G2', 'G3_m')}}
    runs = [{'result': good}, {'result': {**good, 'status': 'incomplete'}}]
    assert not compare(runs, images, 1e-3, .01)['budget_convergence_passed']
    runs[1]['result'] = good
    assert compare(runs, images, 1e-3, .01)['budget_convergence_passed']
    runs[1]['result'] = {**good, 'G1': {'g': float('nan')}}
    assert not compare(runs, images, 1e-3, .01)['budget_convergence_passed']


def test_family_requires_every_seed_regime_crop_exactly_once():
    rows = [{'seed': s, 'dr0': d, 'crop': c} for s in (1, 2) for d in (4., 8.)
            for c in ('feature', 'bland')]
    assert complete_family(rows, (1, 2), (4., 8.))
    assert not complete_family(rows[:-1], (1, 2), (4., 8.))
    assert not complete_family(rows+rows[:1], (1, 2), (4., 8.))
    assert not complete_family(rows[:-1]+rows[:1], (1, 2), (4., 8.))
