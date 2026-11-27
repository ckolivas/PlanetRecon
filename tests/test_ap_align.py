"""AutoStakkert-style alignment points, local quality and their stacking integration."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline.ap_align import AlignmentPoints, LocalQuality, select_locally
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.local_align import pull
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.reconstruction import ReconstructionConfig


def planet(shape=(240, 320), seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape, dtype=float)
    disc = np.hypot(yy - shape[0]/2, xx - shape[1]/2) < .42*min(shape)
    texture = gaussian_filter(rng.normal(size=shape), 3)
    return gaussian_filter(disc*(100 + 40*texture/texture.std()), 1.)


def warp(image, fx, fy, shift=(0., 0.)):
    yy, xx = np.indices(image.shape, dtype=float)
    # Observed frame(x) = object(x - f - g); the sampling field that restores it is f + g.
    return map_coordinates(image, [yy - fy - shift[1], xx - fx - shift[0]], order=3, mode='constant')


def test_local_field_and_global_shift_recovered_at_full_strength():
    ref = planet()
    yy, xx = np.indices(ref.shape, dtype=float)
    fx, fy = 1.5*np.sin(xx/50), -1.2*np.cos(yy/45)
    shift = (3.25, -2.5)
    frame = warp(ref, fx, fy, shift) + np.random.default_rng(1).normal(0, 1, ref.shape)
    matcher = AlignmentPoints(ref, size=48)
    ux, uy = matcher.displacement(frame, shift)
    inner = np.hypot(yy - 120, xx - 160) < 70
    assert len(matcher.centres) > 20 and matcher.acceptance() > .9
    assert np.sqrt(np.mean((ux - fx - shift[0])[inner]**2)) < .2
    assert np.sqrt(np.mean((uy - fy - shift[1])[inner]**2)) < .2
    # Correcting with the field restores the reference far better than global motion.
    roi = np.s_[60:-60, 80:-80]
    assert (np.linalg.norm((pull(frame, (ux, uy)) - ref)[roi])
            < .35*np.linalg.norm((pull(frame, shift) - ref)[roi]))


def test_straight_edges_constrain_only_the_perpendicular_direction():
    from planetrecon.pipeline.ap_align import _peak_fit
    from planetrecon.pipeline.local_align import LocalRegistration
    yy, xx = np.indices((200, 260), dtype=float)
    stripes = 100 + 50*np.sin(yy/4)*(np.abs(xx - 130) < 110)*(np.abs(yy - 100) < 85)
    frame = warp(stripes, np.full(stripes.shape, 1.), np.full(stripes.shape, .6))
    matcher = AlignmentPoints(stripes, size=48)
    offset, H, ok, _ = _peak_fit(matcher._ncc(LocalRegistration.proxy(frame)))
    interior = ok & (np.abs(matcher.centres[:, 1] - 130) < 50)
    assert interior.sum() >= 3
    # Points inside the stripes carry no information about motion along them.
    assert (H[interior, 0, 0] < 1e-3*H[interior, 1, 1]).all()
    np.testing.assert_allclose(offset[interior, 1], .6, atol=.1)
    ux, uy = matcher.displacement(frame, (0., 0.))
    assert abs(np.median(uy[60:140, 70:190]) - .6) < .1


def test_untextured_or_dark_capture_keeps_global_motion():
    flat = np.full((160, 200), 100.)
    matcher = AlignmentPoints(flat, size=48)
    assert not matcher.layers
    ux, uy = matcher.displacement(flat, (1.5, -.5))
    assert np.all(ux == 1.5) and np.all(uy == -.5)
    with pytest.raises(ValueError, match='alignment point size'):
        AlignmentPoints(flat, size=8)


def test_brightness_change_is_not_motion():
    ref = planet()
    ux, uy = AlignmentPoints(ref, size=48).displacement(1.3*ref + 20, (0., 0.))
    assert np.abs(ux).max() < .02 and np.abs(uy).max() < .02


def test_local_quality_keeps_each_regions_own_sharpest_frames():
    ref = planet()
    matcher = AlignmentPoints(ref, size=48)
    left = np.indices(ref.shape)[1] < ref.shape[1]//2
    blurred = gaussian_filter(ref, 2.5)
    frames = {0: np.where(left, ref, blurred), 1: np.where(left, blurred, ref)}
    quality = LocalQuality(matcher)
    q0, q1 = (quality.measure(frames[i], (0., 0.)) for i in (0, 1))
    on_left = matcher.centres[:, 1] < ref.shape[1]//2 - 30
    on_right = matcher.centres[:, 1] > ref.shape[1]//2 + 30
    assert (q0 > q1)[on_left].all() and (q1 > q0)[on_right].all()
    chosen = select_locally(matcher, np.array([0, 1]), lambda i: frames[i], lambda r, p: (0., 0.),
                            ref, 1, np.array([True, False]))
    w0, w1 = chosen.weights(0), chosen.weights(1)
    np.testing.assert_allclose(w0 + w1, 1., atol=1e-12)
    assert w0[120, 60] > .99 and w1[120, 260] > .99
    assert chosen.frames().all()


@pytest.mark.skipif(not __import__('importlib').util.find_spec('torch'), reason='torch unavailable')
def test_cuda_field_matches_cpu():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    ref = planet()
    yy, xx = np.indices(ref.shape, dtype=float)
    frame = warp(ref, np.sin(xx/40), .8*np.cos(yy/35), (1.2, .4))
    matcher = AlignmentPoints(ref, size=48)
    cpu = matcher.displacement(frame, (1.2, .4))
    gpu = matcher.displacement_tensor(frame, (1.2, .4)).cpu().numpy()
    np.testing.assert_allclose(gpu, np.array(cpu), atol=1e-9)
    assert matcher.provenance['cuda_execution']['used']
    np.testing.assert_allclose(LocalQuality(matcher, use_cuda=True).measure(frame, (1.2, .4)),
                               LocalQuality(matcher).measure(frame, (1.2, .4)), rtol=1e-10)
    from planetrecon.backends.torch_ap import gaussian_proxy
    from planetrecon.pipeline.local_align import LocalRegistration
    np.testing.assert_allclose(gaussian_proxy(torch.as_tensor(frame, device='cuda:0')).cpu().numpy(),
                               LocalRegistration.proxy(frame), atol=1e-10)


def capture(tmp_path, n=24, color_id=0):
    ref = planet((128, 160), seed=7)*20
    yy, xx = np.indices(ref.shape, dtype=float)
    rng = np.random.default_rng(5)
    frames = []
    for i in range(n):
        f = warp(ref, .9*np.sin(yy/18 + i), .7*np.cos(xx/22 - i), rng.normal(0, 1, 2))
        # Alternate which half is sharp so local quality has a choice to make.
        sharp = (xx < 80) if i % 2 else (xx >= 80)
        f = np.where(sharp, f, gaussian_filter(f, 1.5 + .1*i))
        frames.append(f + rng.normal(0, 5, f.shape))
    frames = np.clip(frames, 0, 4000).astype('u2')
    if color_id == 8:
        frames[:, ::2, ::2] = frames[:, ::2, ::2]*.8
    return write_ser(tmp_path/'ap.ser', frames, color_id=color_id)


def points_config(**kwargs):
    return ReconstructionConfig(**{**dict(device='cpu', threads=2, batch_frames=4, local_alignment=True,
                                          ap_size=32,
                                          stack_percent=50, frame_selection_mode='frame_count'), **kwargs})


@pytest.mark.parametrize('color_id', [0, 8])
def test_stacking_with_alignment_points_and_local_quality(tmp_path, color_id):
    with SERSource(capture(tmp_path, color_id=color_id)) as source:
        cfg = points_config()
        selected = preprocess_source(source, cfg)
        result = stack_source(source, cfg)
        detail = result.provenance['local_alignment']
        assert detail['method'] == 'alignment_points' and detail['points'] > 4
        count = (int(selected.accepted.sum())*50 + 99)//100
        assert detail['local_quality']['frames_per_point'] == count
        # Local selection reads frames outside the global best half.
        assert count < result.n_used <= selected.accepted.sum()
        assert 'alignment points' in result.provenance['registration']
        assert np.isfinite(result.image).all() and result.validity.any()
        # Every screened frame is a candidate.
        assert detail['local_quality']['ranked_frames'] == selected.accepted.sum()
        # With every screened frame selected each point would use every frame:
        # without Fourier weighting ranking is skipped and the result equals
        # plain alignment-point stacking; Fourier weighting ranks for its bins.
        everything = replace(cfg, stack_percent=100, fourier_weighting=False)
        skipped = stack_source(source, everything)
        assert skipped.provenance['local_alignment']['local_quality']['ranked_frames'] == 0
        assert skipped.n_used == selected.accepted.sum()
        binned = stack_source(source, replace(everything, fourier_weighting=True))
        assert binned.provenance['local_alignment']['local_quality']['ranked_frames'] == selected.accepted.sum()
        assert binned.n_used == selected.accepted.sum() and 'fourier_weighting' in binned.provenance


def test_alignment_point_checkpoints_resume_exactly_and_bind_settings(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        cfg = points_config()
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        stop = False

        def event(result, info):
            nonlocal stop
            stop = stop or (result.stage != 'setup' and info['n_processed'] >= 8)
        checkpoint = tmp_path/'points.npz'
        partial = stack_source(source, cfg, state_checkpoint=checkpoint, on_event=event,
                               should_cancel=lambda: stop)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        for changed in (replace(cfg, ap_size=48), replace(cfg, local_alignment=False)):
            with pytest.raises(ValueError, match='identity'):
                stack_source(source, changed, resume_from=checkpoint)


def test_alignment_point_settings_and_saved_jobs(tmp_path):
    with pytest.raises(ValueError, match='ap_size'):
        points_config(ap_size=300)
    # Saved jobs from before local quality became unconditional still load,
    # unless they explicitly disabled it.
    loaded = ReconstructionConfig.from_dict({**points_config().to_dict(), 'local_quality': True,
                                             'local_quality_pool': 2.})
    assert loaded == points_config()
    with pytest.raises(ValueError, match='local quality was removed'):
        ReconstructionConfig.from_dict({**points_config().to_dict(), 'local_quality': False})
    # Saved jobs naming alignment points (or no local alignment) load without the
    # removed matcher settings; jobs that chose a removed matcher are refused.
    removed = dict(alignment_method='square', local_patch_size=65, sampling_multiplier=5.,
                   alignment_wavelength_nm=550.)
    for method in ('alignment_points', 'square', 'circular_multiscale'):
        saved = {**points_config().to_dict(), **removed, 'alignment_method': method}
        if method == 'alignment_points':
            assert ReconstructionConfig.from_dict(saved) == points_config()
        else:
            with pytest.raises(ValueError, match='Square and circular local alignment were removed'):
                ReconstructionConfig.from_dict(saved)
            plain = ReconstructionConfig.from_dict({**saved, 'local_alignment': False})
            assert plain == replace(points_config(), local_alignment=False)
    # Settings that only alignment points use do not disturb global-alignment checkpoints.
    from planetrecon import resume
    other = ReconstructionConfig(device='cpu')
    with SERSource(capture(tmp_path)) as source:
        assert 'ap_size' not in resume.identity(source, other, None)['config']


def test_cli_alignment_point_inputs_reach_the_config(tmp_path, monkeypatch):
    from planetrecon import cli
    seen = {}

    def fake(source, config, **kwargs):
        seen['config'] = config
        raise SystemExit(0)
    monkeypatch.setattr('planetrecon.pipeline.baseline.stack_source', fake)
    path = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, points_config())
    with pytest.raises(SystemExit):
        cli.main(['stack', '--path', str(path), '--device', 'cpu', '--out', str(tmp_path/'out'),
                  '--ap-size', '40'])
    cfg = seen['config']
    assert cfg.local_alignment and cfg.ap_size == 40


@pytest.mark.parametrize('color_id', [0, 8])
def test_cuda_stacking_matches_cpu_and_reports_setup_progress(tmp_path, color_id):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    with SERSource(capture(tmp_path, color_id=color_id)) as source:
        cfg = points_config()
        preprocess_source(source, cfg)
        cpu = stack_source(source, cfg)
        phases = []

        def event(result, info):
            if result.stage == 'setup':
                phases.append((info['phase'], info['n_processed'], info['n_total']))
        gpu = stack_source(source, replace(cfg, device='gpu'), on_event=event)
        assert gpu.backend == 'cuda' and gpu.provenance['local_alignment']['cuda_execution']['used']
        assert gpu.n_used == cpu.n_used
        np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-7, atol=1e-7*cpu.image.max())
        np.testing.assert_allclose(gpu.coverage, cpu.coverage, rtol=1e-7, atol=1e-9)
        names = {name for name, _, _ in phases}
        assert names == {'Building alignment reference', 'Ranking frames by local quality', 'Finding moons'}
        assert all(done == total for name, done, total in phases[-1:])


@pytest.mark.registration_precision
@pytest.mark.parametrize('color_id', [0, 8])
def test_float32_device_registration_stays_close_to_the_cpu(tmp_path, color_id):
    """Production float32 matching and footprint blending move the stack far less than its noise."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    import planetrecon.backends.precision as precision
    assert precision.REGISTRATION == precision.BLEND == torch.float32
    with SERSource(capture(tmp_path, color_id=color_id)) as source:
        cfg = replace(points_config(), fourier_weighting=True, stack_percent=50)
        preprocess_source(source, cfg)
        cpu = stack_source(source, cfg)
        gpu = stack_source(source, replace(cfg, device='gpu'))
    execution = gpu.provenance['local_alignment']['cuda_execution']
    assert execution['registration_precision'] == execution['blend_precision'] == 'float32'
    assert gpu.n_used == cpu.n_used
    assert np.abs(gpu.image - cpu.image).max() <= 1e-4 * cpu.image.max()
    # Coverage sums bilinear (CFA) sample weights at positions moved by ~1e-4 px: a few
    # pixels where those weights are steep move by up to ~1e-3.
    relative = np.abs(gpu.coverage - cpu.coverage) / np.maximum(cpu.coverage, 1e-12 * cpu.coverage.max())
    assert np.median(relative) < 2e-5 and np.mean(relative > 1e-4) < 2e-3 and relative.max() < 2e-3


@pytest.mark.hardware
@pytest.mark.parametrize('color_id', [0, 8])
@pytest.mark.parametrize('failure_stage', ['matching', 'backprojection'])
def test_midrun_cuda_failure_continues_on_cpu(tmp_path, monkeypatch, color_id, failure_stage):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.backends.torch_accel import TorchBackend
    from planetrecon.backends.torch_ap import TorchAlignmentPoints
    with SERSource(capture(tmp_path, color_id=color_id)) as source:
        cfg = points_config()
        preprocess_source(source, cfg)
        cpu = stack_source(source, cfg)
        from planetrecon.backends.torch_ap import fused_kernels
        calls = 0

        def failing(original):
            def failure(*args):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('injected CUDA failure')
                return original(*args)
            return failure
        if failure_stage == 'backprojection':
            # Grouped stacking resamples with the fused kernel, else the torch group pull.
            kernels = fused_kernels()
            if kernels is not None:
                monkeypatch.setattr(kernels, 'pull', failing(kernels.pull))
            monkeypatch.setattr(TorchBackend, 'backproject_group', failing(TorchBackend.backproject_group))
        else:
            monkeypatch.setattr(TorchAlignmentPoints, 'displacements', failing(TorchAlignmentPoints.displacements))
        fallback = stack_source(source, replace(cfg, device='gpu'))
    assert calls >= 3
    assert fallback.backend == 'cpu' and any('injected CUDA' in w for w in fallback.warnings)
    assert fallback.n_used == cpu.n_used
    np.testing.assert_allclose(fallback.image, cpu.image, rtol=1e-7, atol=1e-7*cpu.image.max())
    np.testing.assert_allclose(fallback.coverage, cpu.coverage, rtol=1e-7, atol=1e-9)


def test_direction_trust_drops_offsets_chosen_by_noise():
    from planetrecon.pipeline.ap_align import direction_trust, noise_spread, trusted
    rng = np.random.default_rng(4)
    n = 400
    offsets = rng.normal(0, 1.5, (n, 3, 2))
    # Point 1 lies on a ridge along x: noise picks its x anywhere in the search window.
    offsets[:, 1, 0] = rng.uniform(-5.5, 5.5, n)
    information = np.tile(np.diag([.02, .02]), (n, 3, 1, 1))
    information[:, 1] = np.diag([.001, .02])
    ok = np.ones((n, 3), bool)
    ok[10:, 2] = False  # Too few accepted fits to judge: unchanged.
    trust, summary = direction_trust(offsets, information, ok)
    assert noise_spread() == pytest.approx(11 / np.sqrt(12))
    assert summary['seeing_spread_px'] == pytest.approx(1.5, abs=.15) and summary['points_measured'] == 2
    np.testing.assert_allclose(trust[0], np.eye(2), atol=.1)
    np.testing.assert_allclose(trust[1], np.diag([0., 1.]), atol=.1)
    np.testing.assert_array_equal(trust[2], np.eye(2))
    kept = trusted(information[0], trust)
    assert kept[1, 0, 0] < .1 * information[0, 1, 0, 0] and kept[1, 1, 1] == pytest.approx(.02, rel=.1)
    assert summary['directions_mostly_noise'] == 1


def banded(shape=(200, 260)):
    """A disc with belts only: nothing to match along them."""
    yy, xx = np.indices(shape, dtype=float)
    disc = np.hypot((yy - shape[0]/2)/.9, xx - shape[1]/2) < 100
    return gaussian_filter(disc*(100 + 30*np.sin(yy/3.)), 1.5)


def test_belts_do_not_let_noise_choose_motion_along_them():
    ref = banded()
    rng = np.random.default_rng(7)
    frames = [ref + rng.normal(0, 6, ref.shape) for _ in range(160)]
    matcher = AlignmentPoints(ref + rng.normal(0, 1, ref.shape), size=48)
    inner = np.flatnonzero(np.hypot(matcher.centres[:, 0] - 100, matcher.centres[:, 1] - 130) < 45)
    assert len(inner) >= 4

    def along_belts():
        return np.std([matcher.displacement(frame, (0., 0.))[0][70:130, 100:160].mean() for frame in frames[:40]])
    untrusted = along_belts()
    matcher.calibrate((frame, (0., 0.)) for frame in frames)
    detail = matcher.provenance['direction_trust']
    assert detail['calibration_frames'] == 160 and detail['directions_mostly_noise'] >= len(inner)
    # Inside the belts the across-belt direction is kept and the along-belt one dropped.
    np.testing.assert_allclose(np.abs(matcher.trust[inner, 1, 1]), 1., atol=.15)
    assert np.all(np.abs(matcher.trust[inner, 0, 0]) < .5)
    assert along_belts() < .25 * untrusted
    import torch
    if torch.cuda.is_available():
        np.testing.assert_allclose(matcher.displacement_tensor(frames[0], (.3, -.2)).cpu().numpy(),
                                   np.array(matcher.displacement(frames[0], (.3, -.2))), atol=1e-9)
    # A predicted reference on the same grid takes the calibrated trust by centre.
    copy = AlignmentPoints(ref, size=48, trust=matcher.trust_map())
    lookup = matcher.trust_map()
    for centre, t in zip(copy.centres.tolist(), copy.trust):
        np.testing.assert_array_equal(t, lookup.get(tuple(centre), np.eye(2)))
