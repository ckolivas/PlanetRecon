"""Alignment points and local quality with the Surface and Saturn motion models."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import OblateGlobeModel
from planetrecon.geometry.pose import FramePose
from test_globe_registration import sphere, config
from test_motion_local import selected


def smooth_field(shape, seed, amplitude, sigma=12):
    rng = np.random.default_rng(seed)
    fields = [gaussian_filter(rng.normal(size=shape), sigma) for _ in range(2)]
    return tuple(amplitude*f/np.abs(f).max() for f in fields)


def test_rotated_frame_seeing_is_corrected_in_frame_coordinates():
    """Reference predicted at the frame time, points on its observed support, scatter once."""
    from planetrecon.geometry.model import render_observed
    from planetrecon.geometry.warp import bilinear_push
    from planetrecon.pipeline.motion_local import local_coordinates
    shape = (160, 192)
    y, x = np.indices(shape, dtype=float)
    disc = np.hypot(y - 80, x - 96) < 60
    texture = gaussian_filter(np.random.default_rng(4).normal(size=shape), 2.)
    template = gaussian_filter(disc*(300 + 600*texture/texture.std()), 1.)
    model = OblateGlobeModel(GlobeParams(60, surface_rate_rad_s=.05, sub_obs_lat_rad=.2), apply_field=False)
    anchor, pose = FramePose(0., 0., 96., 80.), FramePose(2., 0., 97.4, 79.2)
    render = lambda image, src, ref: render_observed(image, model, src, ref)
    predicted = render(template, pose, anchor)        # seeing-free frame at its own time
    prepared = np.stack((template, np.ones(shape)), axis=-1)
    identity = local_coordinates(prepared, predicted, pose, anchor, render, 33, False, method='alignment_points',
                                 ap_size=32)
    np.testing.assert_allclose(identity[0], x + .5, atol=1e-6)
    np.testing.assert_allclose(identity[1], y + .5, atol=1e-6)
    ux, uy = smooth_field(shape, 5, 1.6, sigma=25)
    observed = map_coordinates(predicted, [y + uy, x + ux], order=3, mode='constant')
    qx, qy = local_coordinates(prepared, observed, pose, anchor, render, 33, False, method='alignment_points',
                               ap_size=32)
    total, weight = bilinear_push(observed, qy - .5, qx - .5, shape)
    corrected = np.divide(total, weight, out=np.zeros(shape), where=weight > 0)
    roi = np.hypot(y - 80, x - 96) < 40
    assert (np.linalg.norm((corrected - predicted)[roi])
            < .4*np.linalg.norm((observed - predicted)[roi]))


def test_cuda_alignment_point_motion_matches_cpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.geometry.model import render_observed
    from planetrecon.pipeline.motion_local import local_coordinates
    shape = (128, 160)
    y, x = np.indices(shape, dtype=float)
    template = gaussian_filter((np.hypot(y - 64, x - 80) < 50)*(300 + 200*gaussian_filter(
        np.random.default_rng(9).normal(size=shape), 2.)), 1.)
    model = OblateGlobeModel(GlobeParams(50, surface_rate_rad_s=.04, sub_obs_lat_rad=.1), apply_field=False)
    anchor, pose = FramePose(0., 0., 80., 64.), FramePose(1.5, 0., 80.6, 63.7)
    render = lambda image, src, ref: render_observed(image, model, src, ref)
    ux, uy = smooth_field(shape, 3, 1.2, sigma=20)
    observed = map_coordinates(render(template, pose, anchor), [y + uy, x + ux], order=3, mode='constant')
    prepared = np.stack((template, np.ones(shape)), axis=-1)
    args = (prepared, observed, pose, anchor, render, 33)
    cpu = local_coordinates(*args, False, method='alignment_points', ap_size=32)
    gpu = local_coordinates(*args, True, method='alignment_points', ap_size=32)
    for a, b in zip(cpu, gpu):
        np.testing.assert_allclose(a, b, atol=1e-8)


def seeing_capture(tmp_path, n=8):
    from planetrecon.io.ser import write_ser
    y, x = np.indices((128, 128), dtype=float)
    frames = []
    for i in range(n):
        ux, uy = smooth_field((128, 128), 10 + i, 1.2)
        frames.append(map_coordinates(sphere(i, rate=.04), [y + uy, x + ux], order=3, mode='constant'))
    frames = np.clip(np.stack(frames)*10, 0, 65535).astype('u2')  # spline undershoot at the limb
    return write_ser(tmp_path/'surface.ser', frames, timestamps=np.arange(n)*10000000)


def points_config(**kwargs):
    return replace(config(rate=.04), frame_preselection=True, local_alignment=True,
                   alignment_method='alignment_points', ap_size=32, stack_percent=50,
                   frame_selection_mode='frame_count', exposure_s=0., batch_frames=2, **kwargs)


def test_surface_alignment_points_correct_seeing_with_local_quality_and_resume(tmp_path):
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    cfg = points_config()
    with SERSource(seeing_capture(tmp_path)) as src:
        selection = selected(src, cfg)
        whole = stack_source(src, cfg, preprocessing=selection)
        detail = whole.provenance['local_alignment']
        assert detail['method'] == 'alignment_points' and detail['points'] > 2
        assert detail['local_quality']['ranked_frames'] == 8
        assert detail['local_quality']['frames_per_point'] == 4
        # Each point keeps its own best frames, so more than the global four contribute.
        assert 4 < whole.n_used <= 8
        ordinary = stack_source(src, replace(cfg, local_alignment=False), preprocessing=selection)
        assert ordinary.n_used == 4 and not np.array_equal(whole.image, ordinary.image)
        state = tmp_path/'state.npz'
        stop = False

        def event(result, info):
            nonlocal stop
            stop = stop or (result.stage not in ('cache_ready', 'setup') and info['n_processed'] >= 3)
        partial = stack_source(src, cfg, preprocessing=selection, on_event=event,
                               should_cancel=lambda: stop, state_checkpoint=state)
        assert partial.incomplete
        resumed = stack_source(src, cfg, preprocessing=selection, resume_from=state)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        with pytest.raises(ValueError, match='identity'):
            stack_source(src, replace(cfg, ap_size=48), preprocessing=selection, resume_from=state)


def test_alignment_points_refuse_field_motion(tmp_path):
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    with SERSource(seeing_capture(tmp_path)) as src:
        with pytest.raises(ValueError, match='Alignment points support Motion model None, Surface, Combined or Saturn'):
            stack_source(src, points_config(geometry_mode='field'))


def saturn_seeing_capture(tmp_path, n=8):
    """Rotating globe, field rotation and camera drift with smooth seeing (test_cuda_saturn geometry)."""
    from planetrecon.geometry.globe import GlobeParams
    from planetrecon.geometry.rings import RingParams
    from planetrecon.geometry.saturn import render_saturn
    from planetrecon.io.ser import write_ser
    from test_cuda_saturn import capture
    _, cfg = capture('mono')
    globe = GlobeParams(20., .1, .3, .4, 0., .07, 1.5)
    rings = RingParams(26., 42., .35, .8, .2)
    texture = lambda lon, lat: 1 + .2*np.cos(5*lon)*np.cos(3*lat) + .1*np.sin(9*lon + 4*lat)
    y, x = np.indices((112, 112), dtype=float)
    drift = np.random.default_rng(31).uniform(-1.5, 1.5, size=(n, 2))
    frames = []
    for t in range(n):
        image = render_saturn(112, 112, FramePose(t*.5, .025*(t*.5 - 1.5) + .1, 56 + drift[t, 0], 56 + drift[t, 1]),
                              globe, rings, texture, field_angle0_rad=.1)
        ux, uy = smooth_field((112, 112), 40 + t, 1., sigma=10)
        frames.append(map_coordinates(image, [y + uy, x + ux], order=3, mode='constant'))
    frames = np.clip(np.stack(frames)*1000, 0, 65535).astype('u2')
    path = write_ser(tmp_path/'saturn.ser', frames, timestamps=np.arange(n)*5000000)
    return path, replace(cfg, frame_preselection=True, local_alignment=True, alignment_method='alignment_points',
                         ap_size=24, stack_percent=50, frame_selection_mode='frame_count')


def test_saturn_alignment_points_with_local_quality_and_resume(tmp_path):
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    path, cfg = saturn_seeing_capture(tmp_path)
    with SERSource(path) as src:
        selection = selected(src, cfg)
        whole = stack_source(src, cfg, preprocessing=selection)
        detail = whole.provenance['local_alignment']
        assert whole.provenance['geometry']['registration'].startswith('exposed rings')
        assert detail['method'] == 'alignment_points' and detail['points'] > 2
        assert detail['local_quality']['ranked_frames'] == 8
        assert 4 < whole.n_used <= 8
        ordinary = stack_source(src, replace(cfg, local_alignment=False), preprocessing=selection)
        assert ordinary.n_used == 4 and not np.array_equal(whole.image, ordinary.image)
        for layer in ('globe', 'ring'):
            assert whole.layer_coverage[layer].max() > 0
        state = tmp_path/'state.npz'
        stop = False

        def event(result, info):
            nonlocal stop
            stop = stop or (result.stage not in ('cache_ready', 'setup') and info['n_processed'] >= 3)
        partial = stack_source(src, cfg, preprocessing=selection, on_event=event,
                               should_cancel=lambda: stop, state_checkpoint=state)
        assert partial.incomplete
        resumed = stack_source(src, cfg, preprocessing=selection, resume_from=state)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)


def test_cuda_saturn_alignment_point_stack_matches_cpu(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    from test_cuda_saturn import equal_result
    path, cfg = saturn_seeing_capture(tmp_path)
    with SERSource(path) as src:
        selection = selected(src, cfg)
        cpu = stack_source(src, cfg, preprocessing=selection)
        gpu = stack_source(src, replace(cfg, device='gpu'), preprocessing=selection)
    assert gpu.backend == 'cuda'
    equal_result(gpu, cpu)
    for layer in ('globe', 'ring'):
        np.testing.assert_allclose(gpu.layer_coverage[layer], cpu.layer_coverage[layer], rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_cuda_surface_alignment_point_stack_matches_cpu(tmp_path, colour):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.io.ser import SERSource, write_ser, NAME_TO_COLOR
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.detector import cfa_labels
    path = seeing_capture(tmp_path)
    if colour == 'RGGB':
        with SERSource(path) as src:
            frames = np.stack([src.read_raw(i) for i in range(src.n_frames())]).astype(float)
        labels = cfa_labels(128, 128, colour)
        frames *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
        path = write_ser(tmp_path/'colour.ser', frames.astype('u2'), color_id=NAME_TO_COLOR[colour],
                         timestamps=np.arange(len(frames))*10000000)
    cfg = points_config()
    with SERSource(path) as src:
        selection = selected(src, cfg)
        cpu = stack_source(src, cfg, preprocessing=selection)
        gpu = stack_source(src, replace(cfg, device='gpu'), preprocessing=selection)
    assert gpu.backend == 'cuda' and gpu.n_used == cpu.n_used
    np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-7, atol=1e-6*cpu.image.max())
    np.testing.assert_allclose(gpu.coverage, cpu.coverage, rtol=1e-7, atol=1e-9)


@pytest.mark.registration_precision
@pytest.mark.parametrize('kind', ['surface', 'saturn'])
def test_float32_registration_stays_close_to_float64(tmp_path, kind):
    """Production device registration (float32) moves stacks far less than their noise."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.baseline import stack_source
    import planetrecon.backends.precision as precision
    assert precision.REGISTRATION == torch.float32
    if kind == 'surface':
        path, cfg = seeing_capture(tmp_path), points_config()
    else:
        path, cfg = saturn_seeing_capture(tmp_path)
    with SERSource(path) as src:
        selection = selected(src, cfg)
        cpu = stack_source(src, cfg, preprocessing=selection)
        gpu = stack_source(src, replace(cfg, device='gpu'), preprocessing=selection)
    assert gpu.provenance['geometry_execution']['registration_precision'] == 'float32'
    assert gpu.provenance['geometry_execution']['accumulator_precision'] == 'float64'
    assert gpu.n_used == cpu.n_used
    difference = np.abs(gpu.image - cpu.image)
    assert difference.max() <= 1e-4 * cpu.image.max()
    # Coverage sums bilinear weights at float32 sample positions.
    np.testing.assert_allclose(gpu.coverage, cpu.coverage, rtol=1e-4, atol=1e-6)
