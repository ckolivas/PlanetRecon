"""Assessment frames cannot influence the object or the initialization choice."""
from copy import deepcopy

import numpy as np
import pytest

from planetrecon.config import make_config
from planetrecon.mfbd import PupilForward, assessment_split, assess_selected_fit


@pytest.mark.parametrize('n', [3, 8, 20, 100])
def test_three_way_partitions_are_disjoint_nonempty_and_reproducible(n):
    parts = assessment_split(n, .1, 1001)
    assert all(len(p) for p in parts)
    assert sorted(np.concatenate(parts)) == list(range(n))
    assert len(np.unique(np.concatenate(parts))) == n
    for a, b in zip(parts, assessment_split(n, .1, 1001)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('n,frac', [(2, .1), (8, 0), (8, .5), (8, float('nan'))])
def test_invalid_three_way_design_is_rejected(n, frac):
    with pytest.raises(ValueError):
        assessment_split(n, frac, 1001)


def fixture():
    fwd = PupilForward.from_config(make_config(1001, 4., n_diam=8, pupil_pad_factor=8., eval_size=16), n_modes=3)
    y, x = np.indices((16, 16))
    obj = np.exp(-((x-8)**2+(y-8)**2)/10.)
    alpha = np.zeros((4, 3))
    images = np.fft.ifft2(fwd.otfs(alpha)*np.fft.fft2(obj)).real
    fit = {'object': obj, 'alphas': alpha, 'train_idx': np.array([0, 1]), 'holdout_idx': np.array([2]),
           'stages': [{'M': 3, 'object_info': {'converged': True}, 'phase_fits': []}]}
    return fwd, fit, images, alpha


def test_assessment_freezes_object_and_only_fits_new_frames(monkeypatch):
    from planetrecon import mfbd
    fwd, fit, images, alpha = fixture()
    old = deepcopy(fit)
    images[3] += .1
    calls = []
    original = mfbd.fit_frame_alpha

    def record(a, forward, obj_f, image, variance, **kwargs):
        calls.append(image.copy())
        np.testing.assert_allclose(np.fft.ifft2(obj_f).real, old['object'], atol=1e-14)
        return original(a, forward, obj_f, image, variance, **kwargs)

    monkeypatch.setattr(mfbd, 'fit_frame_alpha', record)
    result = assess_selected_fit(fwd, fit, images, np.ones(4), [3], alpha, alpha_iters=2)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], images[3])
    np.testing.assert_array_equal(fit['object'], old['object'])
    np.testing.assert_array_equal(fit['alphas'], old['alphas'])
    assert not np.any(alpha)
    assert result['indices'] == [3] and result['loss'] > 0
    assert 'not unfitted prediction' in result['metric_role']


@pytest.mark.parametrize('indices', [[], [0], [2], [3, 3], [-1], [4]])
def test_assessment_rejects_leaked_or_invalid_frames(indices):
    fwd, fit, images, alpha = fixture()
    with pytest.raises(ValueError):
        assess_selected_fit(fwd, fit, images, np.ones(4), indices, alpha)


def test_assessment_cannot_make_unconverged_training_valid():
    fwd, fit, images, alpha = fixture()
    fit['stages'][0]['object_info']['converged'] = False
    result = assess_selected_fit(fwd, fit, images, np.ones(4), [3], alpha)
    assert result['loss'] < 1e-20
    assert result['status'] == 'incomplete'


def test_q2_selection_is_invariant_to_assessment_pixels(monkeypatch):
    from types import SimpleNamespace
    from planetrecon import q2

    n, size = 8, 4
    train, select, assess = assessment_split(n, .1, 1001)
    crop = SimpleNamespace(name='feature', observed=np.arange(1., n+1)[:, None, None]*np.ones((n, size, size)),
                           expected=np.ones((n, size, size)), shifts=np.zeros((n, 2)),
                           support=np.ones((size, size)), sky_mask=None)
    known = {key: {'_info': {'converged': True}, 'E_H': 3.}
             for key in ('E2a_S10', 'E2a_S100', 'E2b_S10', 'E2b_S100', 'A1o_S10')}
    known.update(E_H_E2_star=1., E2_star='E2b', prior_limited=False)
    monkeypatch.setattr(q2, '_known_transfer_block', lambda *a: known)
    monkeypatch.setattr(q2, '_metrics', lambda *a: {'E_H': 2., 'high_band_illconditioned': False})
    monkeypatch.setattr(q2.Regularisation, 'field_e2a', lambda *a: np.ones((size, size)))
    monkeypatch.setattr(q2.PupilForward, 'from_config', lambda *a: None)
    monkeypatch.setattr(q2, 'tip_tilt_from_shifts', lambda *a, **k: (np.zeros((n, 2)), {}))
    monkeypatch.setattr(q2, 'score_sequence', lambda images, **k: images.mean(axis=(1, 2)))
    monkeypatch.setattr(q2, 'initial_object', lambda f, images, v, l, s, idx, tv, **k: images[idx].mean(axis=0))

    def fit(f, images, variance, lam, support, **kw):
        obj = kw['obj0'].copy()
        loss = float(obj.mean())
        stage = {'M': 2, 'object': obj, 'alphas': kw['alpha0'], 'otfs': None,
                 'train_loss': loss, 'holdout_loss': loss if kw['holdout_idx'].size else None,
                 'n_outer': 1, 'phase_fits': [], 'object_info': {'converged': True}}
        return {'object': obj, 'alphas': kw['alpha0'], 'stages': [stage],
                'train_idx': kw['train_idx'], 'holdout_idx': kw['holdout_idx']}

    frozen_objects = []

    def assess_fit(f, fit, images, variance, indices, alpha, **kw):
        np.testing.assert_array_equal(fit['train_idx'], train)
        np.testing.assert_array_equal(fit['holdout_idx'], select)
        np.testing.assert_array_equal(indices, assess)
        frozen_objects.append(fit['object'].copy())
        return {'status': 'valid', 'loss': float(images[indices].sum())}

    monkeypatch.setattr(q2, 'd_tail', fit)
    monkeypatch.setattr(q2, 'assess_selected_fit', assess_fit)
    kwargs = dict(holdout=True, m_grid=(2,), outer_iters=(1,), alpha_iters=1, tv_mu=0.)
    before = q2.evaluate_q2_crop(SimpleNamespace(eval_size=size), crop, {'read_noise_e': 1., 'seed': 1001}, **kwargs)
    crop.observed[assess] += 1000.
    after = q2.evaluate_q2_crop(SimpleNamespace(eval_size=size), crop, {'read_noise_e': 1., 'seed': 1001}, **kwargs)
    assert before['chosen_init'] == after['chosen_init']
    assert before['holdout']['holdout_loss'] == after['holdout']['holdout_loss']
    assert before['assessment']['loss'] != after['assessment']['loss']
    np.testing.assert_array_equal(*frozen_objects)
