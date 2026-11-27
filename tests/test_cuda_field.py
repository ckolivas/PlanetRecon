"""Field derotation on CUDA must follow the CPU tracking, projection and accumulation."""
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('CUDA device unavailable', allow_module_level=True)

from planetrecon.backends.torch_accel import TorchBackend
from planetrecon.backends.torch_globe import TorchGlobeWarp
from planetrecon.backends.torch_surface import TorchFieldTracker
from planetrecon.detector import cfa_labels
from planetrecon.geometry.model import FieldOnlyModel, render_observed
from planetrecon.geometry.pose import FramePose
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.field_align import field_displacement
from planetrecon.reconstruction import ReconstructionConfig
from test_field_observed_support import cropped_scene


def disc(angle, dx=0., dy=0., shape=(96, 112), radius=36.):
    """A textured disc inside the frame, turned by ``angle`` about its centre and drifted."""
    h, w = shape
    y, x = np.indices(shape, dtype=float)
    sx, sy = x + .5 - w / 2 - dx, h / 2 + dy - y - .5
    r, theta = np.hypot(sx, sy), np.arctan2(sy, sx) - angle
    texture = 100 + 25 * np.cos(3 * theta) * np.sin(r / 5) + 12 * np.sin(5 * theta + r / 7)
    return np.where(r < radius, texture, 4.)


@pytest.mark.parametrize('shape', [(64, 64), (33, 50), (96, 112)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_per_frame_reference_correlation_matches_the_cpu(shape, dtype):
    from scipy.ndimage import gaussian_filter, shift
    rng = np.random.default_rng(31)
    references = np.stack([gaussian_filter(rng.normal(size=shape), 2.5) * 100 for _ in range(5)])
    frames = np.stack([shift(r, rng.normal(0, 3, 2), order=1, mode='wrap') + rng.normal(0, 2, shape)
                       for r in references])
    refs, imgs = (torch.as_tensor(a, device='cuda').to(dtype) for a in (references, frames))
    expected = [phase_correlation_shift(r, f) for r, f in zip(refs.double().cpu().numpy(),
                                                                imgs.double().cpu().numpy())]
    np.testing.assert_allclose(TorchBackend().phase_correlation_pairs(refs, imgs), expected, atol=1e-9)


def test_masked_matcher_can_admit_every_competing_score():
    """field_displacement's matching (no distinct_peaks) lets broad shoulders compete."""
    from planetrecon.backends.torch_surface import MaskedMatcher, masked_templates, _gaussian
    from planetrecon.pipeline.masked_align import MaskedRegistration
    reference, radius = cropped_scene(0.)
    mask = radius < 60
    mask[:6] = mask[-6:] = False
    mask[:, :6] = mask[:, -6:] = False
    frames = [cropped_scene(0., dx=dx, dy=dy)[0] for dx, dy in ((1.3, -.7), (-2.2, .4), (0., 0.))]
    expected = [MaskedRegistration(reference, mask).displacement(f) for f in frames]
    matcher = MaskedMatcher(reference.shape, distinct=False)
    template, count, energy = masked_templates(matcher.tensor(reference), torch.as_tensor(mask, device='cuda'))
    n = len(frames)
    found = matcher.batch(template.expand(n, -1, -1), torch.as_tensor(mask, device='cuda').expand(n, -1, -1),
                          [float(count)] * n, [float(energy)] * n, _gaussian(matcher.tensor(np.stack(frames)), 1.5))
    for a, b in zip(found, expected):
        assert (a is None) == (b is None)
        if b is not None:
            np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize('static', [False, True])
def test_field_tracker_matches_the_cpu_phase_correlation(static):
    reference_pose = FramePose(0., 0., 56., 48.)
    model = FieldOnlyModel()
    warp = TorchGlobeWarp((96, 112), model)
    tracker = TorchFieldTracker(warp, TorchBackend(), disc(0.), reference_pose, 36., cropped=False, static=static)
    poses = [FramePose(float(k), 0. if static else .03 * k, 56., 48.) for k in range(1, 6)]
    frames = np.stack([disc(p.field_angle_rad, dx=.7 * k - 1.1, dy=1.3 - .4 * k) for k, p in enumerate(poses)])
    expected = [phase_correlation_shift(disc(0.) if static else render_observed(disc(0.), model, p, reference_pose), f)
                for p, f in zip(poses, frames)]
    found = tracker.displacements(torch.as_tensor(frames, device='cuda'), poses)
    np.testing.assert_allclose(found, expected, atol=1e-9)
    np.testing.assert_allclose(tracker.displacement(frames[2], poses[2]), expected[2], atol=1e-9)


def test_cropped_field_tracker_matches_field_displacement():
    reference, _ = cropped_scene(0.)
    reference_pose = FramePose(0., 0., 44., 48.)
    tracker = TorchFieldTracker(TorchGlobeWarp(reference.shape, FieldOnlyModel()), TorchBackend(), reference,
                                reference_pose, 70., cropped=True, static=False)
    poses = [FramePose(1., angle, 44., 48.) for angle in (.01, -.02, .035, .05)]
    frames = np.stack([cropped_scene(p.field_angle_rad, dx=dx, dy=dy)[0]
                       for p, (dx, dy) in zip(poses, ((.6, -.3), (-1.2, .8), (0., 0.), (2.1, 1.4)))])
    frames[3] = 1.  # Featureless: the observed field cannot constrain its drift.
    expected = [field_displacement(reference, f, p, reference_pose, 70.) for p, f in zip(poses, frames)]
    found = tracker.displacements(torch.as_tensor(frames, device='cuda'), poses)
    assert expected[3] is None and found[3] is None
    np.testing.assert_allclose(found[:3], expected[:3], atol=1e-9)


def frames_in(colour, frames):
    if colour == 'RGB':
        return frames[..., None] * [.8, 1., .6]
    if colour != 'mono':
        labels = cfa_labels(*frames.shape[1:], colour)
        return frames * np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
    return frames


@pytest.mark.parametrize('colour', ['mono', 'RGB', 'RGGB', 'BGGR'])
@pytest.mark.parametrize('field_rate', [0., .02])
def test_cuda_field_stack_matches_the_cpu(colour, field_rate):
    drift = [(0., 0.), (1.3, -2.4), (-2., 1.), (.3, .4), (.9, -.6)]
    frames = frames_in(colour, np.stack([disc(field_rate * t, dx, dy) for t, (dx, dy) in enumerate(drift)]))
    source = ArraySource(frames, color_mode=colour, bit_depth=32, timestamps=np.arange(5.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False, geometry_mode='field',
                               field_center_x=56, field_center_y=48, equatorial_radius_px=36,
                               field_rate_rad_s=field_rate)
    cpu = stack_source(source, cfg)
    gpu = stack_source(source, replace(cfg, device='gpu'))
    assert gpu.backend == 'cuda' and not gpu.provenance['device_report']['fallback']
    assert gpu.provenance['geometry_execution']['drift_matching'] == 'cuda'
    assert gpu.n_used == cpu.n_used == 5
    np.testing.assert_array_equal(gpu.validity, cpu.validity)
    np.testing.assert_allclose(gpu.coverage, cpu.coverage, atol=2e-8, rtol=2e-9)
    np.testing.assert_allclose(gpu.image, cpu.image, atol=2e-7, rtol=2e-9)


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_cuda_cropped_field_stack_matches_the_cpu(colour):
    angles = np.linspace(0, np.deg2rad(3), 7)
    drift = [(0., 0.), (.8, -.5), (-1.1, .3), (.4, .9), (0., 0.), (-.6, -.7), (1.2, .2)]
    frames = frames_in(colour, np.stack([cropped_scene(a, dx=dx, dy=dy)[0] * 100
                                         for a, (dx, dy) in zip(angles, drift)]))
    source = ArraySource(frames, color_mode=colour, bit_depth=32, timestamps=np.arange(7.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False, geometry_mode='field',
                               field_center_x=44, field_center_y=48, equatorial_radius_px=70,
                               field_rate_rad_s=angles[-1] / 6)
    cpu = stack_source(source, cfg)
    gpu = stack_source(source, replace(cfg, device='gpu'))
    assert gpu.provenance['geometry']['registration'] == cpu.provenance['geometry']['registration'] == \
        'shared observed field; unresolved tracking stops the run'
    assert gpu.backend == 'cuda' and gpu.n_used == cpu.n_used == 7
    np.testing.assert_array_equal(gpu.validity, cpu.validity)
    np.testing.assert_allclose(gpu.coverage, cpu.coverage, atol=2e-8, rtol=2e-9)
    np.testing.assert_allclose(gpu.image, cpu.image, atol=2e-7, rtol=2e-9)


def test_previews_leave_rank_bin_sums_on_the_device(tmp_path, monkeypatch):
    """Fourier weighting's rank-bin sums are copied for the final image and checkpoints, not for previews."""
    from planetrecon.backends.torch_globe import TorchGlobeAccumulator
    from planetrecon.io.ser import SERSource
    from test_motion_local import selected
    from test_motion_points import field_seeing_capture
    path, cfg = field_seeing_capture(tmp_path)
    cfg = replace(cfg, device='gpu', fourier_weighting=True, batch_frames=2)
    downloads = []
    original = TorchGlobeAccumulator.download_bins
    monkeypatch.setattr(TorchGlobeAccumulator, 'download_bins',
                        lambda self: downloads.append(1) or original(self))
    previews = []
    with SERSource(path) as src:
        selection = selected(src, cfg)
        result = stack_source(src, cfg, preprocessing=selection, on_event=lambda r, info: previews.append(r))
        plain = stack_source(src, cfg, preprocessing=selection)
    assert 'fourier_weighting' in result.provenance and len(previews) > 3
    assert len(downloads) == 2  # One final copy per run.
    np.testing.assert_allclose(result.image, plain.image, rtol=1e-12, atol=1e-12 * plain.image.max())
