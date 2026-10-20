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
                                          alignment_method='alignment_points', ap_size=32,
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
        # ranking is skipped and the result equals plain alignment-point stacking.
        everything = replace(cfg, stack_percent=100)
        skipped = stack_source(source, everything)
        assert skipped.provenance['local_alignment']['local_quality']['ranked_frames'] == 0
        assert skipped.n_used == selected.accepted.sum()


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
        for changed in (replace(cfg, ap_size=48), replace(cfg, alignment_method='circular_multiscale')):
            with pytest.raises(ValueError, match='identity'):
                stack_source(source, changed, resume_from=checkpoint)


def test_alignment_points_refuse_motion_models_and_validate_settings(tmp_path):
    with pytest.raises(ValueError, match='ap_size'):
        points_config(ap_size=300)
    # Saved jobs from before local quality became unconditional still load,
    # unless they explicitly disabled it.
    loaded = ReconstructionConfig.from_dict({**points_config().to_dict(), 'local_quality': True,
                                             'local_quality_pool': 2.})
    assert loaded == points_config()
    with pytest.raises(ValueError, match='local quality was removed'):
        ReconstructionConfig.from_dict({**points_config().to_dict(), 'local_quality': False})
    with SERSource(capture(tmp_path)) as source:
        cfg = points_config(geometry_mode='field', field_rate_rad_s=0.)
        with pytest.raises(ValueError, match='Alignment points support Motion model None, Surface, Combined or Saturn'):
            stack_source(source, cfg)
    # Settings that only alignment points use do not disturb other checkpoints.
    from planetrecon import resume
    other = ReconstructionConfig(device='cpu', local_alignment=True, alignment_method='circular_multiscale')
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
                  '--local-alignment', '--alignment-method', 'alignment_points', '--ap-size', '40'])
    cfg = seen['config']
    assert (cfg.alignment_method, cfg.ap_size) == ('alignment_points', 40)


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
        assert names == {'Building alignment reference', 'Ranking frames by local quality'}
        assert all(done == total for name, done, total in phases[-1:])
