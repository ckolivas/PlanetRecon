"""An independently sampled sphere exposes false drift from missing longitudes."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.detector import cfa_labels
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import OblateGlobeModel, render_observed
from planetrecon.geometry.pose import FramePose
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.globe_align import surface_displacement
from planetrecon.reconstruction import ReconstructionConfig


def sphere(t, rate=.1, dx=0., dy=0., field_rate=0.):
    # Equator-on spherical projection, evaluated directly at each detector pixel.
    # No application renderer or inverse coordinate map generates these frames.
    y, x = np.indices((128, 128), dtype=float)
    sx, sy = x+.5-64-dx, 64+dy-y-.5
    angle = field_rate*t
    u = np.cos(angle)*sx + np.sin(angle)*sy
    v = -np.sin(angle)*sx + np.cos(angle)*sy
    inside = u*u+v*v < 42**2
    z = np.sqrt(np.maximum(42**2-u*u-v*v, 0.))
    lon = np.arctan2(u, z)+rate*t
    lat = np.arcsin(np.clip(v/42, -1., 1.))
    texture = 100+20*np.cos(12*lon)*np.cos(3*lat)+15*np.sin(14*lat+2*lon)
    return np.where(inside, texture, 0.)


def config(rate=.1, field_rate=0.):
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='combined' if field_rate else 'surface',
        field_center_x=64, field_center_y=64, equatorial_radius_px=42,
        surface_rate_rad_s=rate, field_rate_rad_s=field_rate)


def old_displacement(reference, frame, model, pose, reference_pose):
    return phase_correlation_shift(render_observed(reference, model, pose, reference_pose), frame)


@pytest.mark.parametrize('rate', [-.1, .1])
@pytest.mark.parametrize('field_rate', [0., .04])
def test_shared_surface_tracks_drift_without_dark_limb_bias(rate, field_rate):
    reference = sphere(0)
    anchor = FramePose(0, 0, 64, 64)
    model = OblateGlobeModel(GlobeParams(42, surface_rate_rad_s=rate))
    for t in (1, 3, 6):
        pose = FramePose(t, field_rate*t, 64, 64)
        for dx, dy in ((0, 0), (3, -2), (-2.4, 1.3)):
            frame = sphere(t, rate, dx, dy, field_rate)
            fitted = surface_displacement(reference, frame, model, pose, anchor)
            assert fitted is not None
            np.testing.assert_allclose(fitted, (dx, dy), atol=.09)
    old = old_displacement(reference, sphere(6, rate, field_rate=field_rate), model, pose, anchor)
    assert np.linalg.norm(old) > 1.


@pytest.mark.parametrize('colour', ['mono', 'RGB', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('field_rate', [0., .04])
def test_visible_surface_matching_improves_noisy_unsharpened_stack(monkeypatch, colour, field_rate):
    offsets = [(0, 0), (1.3, -2.4), (-3.7, 2.2), (.3, .4), (2.5, -1.2), (0, 0), (-2, 3)]
    rng = np.random.default_rng(822)
    frames = np.array([sphere(t, dx=dx, dy=dy, field_rate=field_rate)
                       for t, (dx, dy) in enumerate(offsets)])
    truth = sphere(0)
    coefficients = np.array([.8, 1., .6])
    if colour == 'RGB':
        frames = frames[..., None]*coefficients
        truth = truth[..., None]*coefficients
    elif colour != 'mono':
        labels = cfa_labels(128, 128, colour)
        frames *= np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
        truth = truth[..., None]*coefficients
    frames += rng.normal(0, .5, frames.shape)
    src = ArraySource(frames, color_mode=colour, bit_depth=32, timestamps=np.arange(7.))
    cfg = config(field_rate=field_rate)
    corrected = stack_source(src, cfg)
    monkeypatch.setattr('planetrecon.pipeline.globe_align.surface_displacement', old_displacement)
    old = stack_source(src, cfg)
    y, x = np.indices((128, 128))
    mask = (x+.5-64)**2+(y+.5-64)**2 < 25**2
    def error(result):
        assert result.n_used == 7 and result.n_rejected == 0
        assert result.validity[mask].all()
        return np.sqrt(np.mean((result.image[mask]-truth[mask])**2))
    assert error(corrected) < .65*error(old)


def test_surface_match_honours_shift_limit():
    src = ArraySource(np.array([sphere(0), sphere(1, dx=7)]), bit_depth=32, timestamps=np.arange(2.))
    result = stack_source(src, replace(config(), max_shift_px=4.))
    assert result.n_used == 1 and result.n_rejected == 1


def test_unobserved_or_flat_surface_cannot_set_camera_drift():
    reference = np.ones((128, 128))
    model = OblateGlobeModel(GlobeParams(42, surface_rate_rad_s=.1))
    assert surface_displacement(reference, reference, model,
        FramePose(1, 0, 64, 64), FramePose(0, 0, 64, 64)) is None


def test_newly_visible_brightness_cannot_pull_reference_surface():
    model = OblateGlobeModel(GlobeParams(42, surface_rate_rad_s=.1))
    anchor, pose = FramePose(0, 0, 64, 64), FramePose(6, 0, 64, 64)
    reference, frame = sphere(0), sphere(6)
    y, x = np.indices(reference.shape)
    u, v = x+.5-64, 64-y-.5
    on_globe = u*u+v*v < 42**2
    longitude = np.arctan2(u, np.sqrt(np.maximum(42**2-u*u-v*v, 0.)))+.6
    newly_visible = on_globe & (np.cos(longitude) < 0)
    assert newly_visible.any()
    original = surface_displacement(reference, frame, model, pose, anchor)
    assert original is not None
    for amplitude in (-1000., 1000.):
        changed = frame.copy()
        changed[newly_visible] += amplitude
        fitted = surface_displacement(reference, changed, model, pose, anchor)
        np.testing.assert_allclose(fitted, original, atol=1e-8)


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_shared_surface_keeps_selected_reference_origin(colour):
    offsets = [(2, -4), (-4, 2), (4, 4), (0, 0), (-2, -2), (2, 2), (-4, -2)]
    frames = np.array([sphere(t, dx=dx, dy=dy) for t, (dx, dy) in enumerate(offsets)])
    steady = np.array([sphere(t) for t in range(7)])
    if colour == 'RGGB':
        labels = cfa_labels(128, 128, colour)
        gains = np.where(labels == 'R', .8, np.where(labels == 'B', .6, 1.))
        frames *= gains
        steady *= gains
    cfg = replace(config(), reference_index=3)
    def run(data):
        return stack_source(ArraySource(data, color_mode=colour, bit_depth=32,
                            timestamps=np.arange(7.)), cfg)
    corrected, stationary = run(frames), run(steady)
    assert corrected.provenance['reference_index'] == 3
    assert corrected.n_used == stationary.n_used == 7
    y, x = np.indices((128, 128))
    mask = (x+.5-64)**2+(y+.5-64)**2 < 25**2
    np.testing.assert_allclose(corrected.image[mask], stationary.image[mask], atol=1e-8)
