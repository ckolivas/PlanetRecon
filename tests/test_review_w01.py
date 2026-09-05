"""Regressions for the W00/W01 implementation review (small CPU fixtures)."""

import numpy as np
import pytest
import os
import subprocess
import sys
from scipy.optimize import LinearConstraint, minimize

from planetrecon.estimators import e2a, project_positivity_support
from planetrecon.operators import linear_convolve_same, spatial_convolve_same
from planetrecon.provenance import (
    certificate_is_current,
    current_method_fingerprint,
    experiment_manifest,
    manifest_errors,
)


def test_dykstra_warm_start_projects_the_new_target():
    support = np.ones((4, 4))
    _, old = project_positivity_support(-np.ones((4, 4)), support)
    target = np.ones((4, 4))
    actual, info = project_positivity_support(target, support, p=old['p'], q=old['q'])
    np.testing.assert_allclose(actual, target, atol=1e-12)
    assert info['converged']


def test_dykstra_matches_independent_small_constrained_quadratic():
    n = 4
    freq = np.fft.fftfreq(n)
    support = (freq[:, None]**2 + freq[None, :]**2 <= 0.26**2).astype(float)
    eye = np.eye(n*n)
    projection = np.stack([
        np.fft.ifft2(np.fft.fft2(v.reshape(n, n))*support).real.ravel()
        for v in eye
    ], axis=1)
    eigenvalues, eigenvectors = np.linalg.eigh(projection)
    basis = eigenvectors[:, eigenvalues > 0.5]
    target = np.random.default_rng(42).normal(size=(n, n))
    coeff = basis.T @ target.ravel()
    oracle = minimize(
        lambda z: 0.5*np.sum((z-coeff)**2), np.zeros_like(coeff),
        jac=lambda z: z-coeff, method='SLSQP',
        constraints=LinearConstraint(basis, 0, np.inf),
        options={'ftol': 1e-13, 'maxiter': 500},
    )
    assert oracle.success
    actual, info = project_positivity_support(target, support, maxiter=5000)
    assert info['converged']
    np.testing.assert_allclose(actual.ravel(), basis @ oracle.x, atol=1e-7)
    fitted, fit_info = e2a(
        np.ones((1, n, n), complex), target[None], np.ones(1),
        np.full((n, n), .03), support,
    )
    np.testing.assert_allclose(fitted.ravel(), basis @ oracle.x / 1.03, atol=1e-7)
    assert fit_info['converged']


def test_projection_stagnation_is_not_feasibility():
    target = np.zeros((16, 16))
    target[8, 8] = 1
    f = np.fft.fftfreq(16)
    support = (f[:, None]**2 + f[None, :]**2 < .15**2)
    _, info = project_positivity_support(target, support, maxiter=1, tol=100)
    assert info['positivity_violation'] > 1e-8
    assert not info['converged']


def test_e2a_identity_has_the_exact_nonnegative_solution():
    target = np.random.default_rng(27).normal(size=(8, 8))
    result, info = e2a(
        np.ones((1, 8, 8), complex), target[None], np.ones(1),
        np.full((8, 8), .03), np.ones((8, 8)),
    )
    np.testing.assert_allclose(result, np.maximum(target/1.03, 0), atol=1e-8)
    assert info['converged'] and info['feasible']
    assert info['kkt_residual'] < 1e-8


@pytest.mark.parametrize('shape', [(4, 4), (4, 5), (5, 4), (5, 5)])
def test_spatial_oracle_handles_even_kernels(shape):
    rng = np.random.default_rng(47)
    image, psf = rng.normal(size=(8, 9)), rng.normal(size=shape)
    np.testing.assert_allclose(
        spatial_convolve_same(image, psf), linear_convolve_same(image, psf), atol=1e-12,
    )


@pytest.mark.parametrize('key,value', [
    ('source_hash', 'stale'), ('config_hash', 'stale'),
    ('estimator_operator_version', 'obsolete'),
    ('wavelength_m', float('nan')), ('exposure_samples_J', float('inf')),
])
def test_current_certificate_rejects_stale_or_nonfinite_fields(key, value):
    current = current_method_fingerprint()
    stale = dict(current, **{key: value})
    assert not certificate_is_current(stale, current)


def test_certificate_cannot_be_current_when_both_are_empty():
    assert not certificate_is_current({}, {})


@pytest.mark.parametrize('entry', [
    {'path': 'x', 'status': 'typo'},
    {'path': 'x', 'status': 'valid', 'gate_passed': 1},
    'not a result object',
])
def test_manifest_validates_nested_results(entry):
    manifest = experiment_manifest('test', status='diagnostic', protocol='test')
    manifest['results'] = [entry]
    assert manifest_errors(manifest)


def test_historical_manifest_keeps_historical_identity(tmp_path):
    from planetrecon.evidence import r9_historical_manifest, write_evidence_manifests
    from planetrecon.provenance import load_manifest

    historical = r9_historical_manifest()
    assert historical['estimator_operator_version'] == '1.0'
    assert historical['code_baseline'] == 'd1b600a'
    assert historical['artifact_hashes']
    write_evidence_manifests(tmp_path)
    assert load_manifest(tmp_path/'manifests/r9-historical.json') == historical


def test_cli_does_not_import_numpy_before_parsing_thread_limit():
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; import planetrecon.cli; assert "numpy" not in sys.modules'],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_thread_limit_is_inherited_by_spawned_workers(monkeypatch):
    from planetrecon.runtime import apply_thread_limits, THREAD_ENV_KEYS
    for key in (*THREAD_ENV_KEYS, 'PLANETRECON_THREADS'):
        monkeypatch.setenv(key, os.environ.get(key, '2'))
    apply_thread_limits(3)
    assert os.environ['PLANETRECON_THREADS'] == '3'


def test_high_rank_correlation_does_not_excuse_unstable_top_subset(monkeypatch):
    from planetrecon import rank
    from planetrecon.validate import check_lowfreq_ranking
    from scipy.stats import spearmanr
    scores = np.arange(80, dtype=float)
    permuted = scores.copy()
    permuted[-16:-8], permuted[-8:] = scores[-8:], scores[-16:-8]
    assert spearmanr(scores, permuted).statistic > .85
    outputs = iter([scores, permuted])
    monkeypatch.setattr(rank, 'score_sequence', lambda *a, **k: next(outputs))
    assert not check_lowfreq_ranking()[0].passed
