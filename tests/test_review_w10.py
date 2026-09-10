"""Independent boundary, masking and serialization regressions for Saturn."""
from dataclasses import replace
import json
import time

import numpy as np
import pytest

from planetrecon.detector import cfa_labels
from planetrecon.geometry.coords import detector_xy_grids
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import RingParams, ring_transmission
from planetrecon.geometry.saturn import (
    LAYER_FAR_RING, LAYER_GLOBE, LAYER_MOON, MoonTrack, SaturnSceneModel,
    moon_mask, render_saturn,
)
from planetrecon.io.ser import write_ser
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.geometry_stack import prepare_geometry
from planetrecon.reconstruction import ReconstructionConfig


def setup(opening=.4, transmission=.35, spin=.5, moon=None):
    globe = GlobeParams(12., flattening=.1, sub_obs_lat_rad=opening, surface_rate_rad_s=spin)
    rings = RingParams(16., 26., transmission=transmission)
    return SaturnSceneModel(globe, rings, moon=moon)


def config(**kwargs):
    cfg = ReconstructionConfig(device="cpu", threads=2, geometry_mode="saturn",
        field_center_x=32., field_center_y=24., equatorial_radius_px=12.,
        flattening=.1, sub_obs_lat_rad=.4, ring_inner_radius_px=16.,
        ring_outer_radius_px=26., field_rate_rad_s=.3, surface_rate_rad_s=.5)
    return replace(cfg, **kwargs)


def test_exposed_far_ring_is_classified_by_depth():
    model = setup()
    x, y = detector_xy_grids(48, 64)
    info = model.classify_detector(x, y, FramePose(0., 0., 32., 24.))
    exposed_far = info["on_ring"] & ~info["on_globe"] & (info["t_ring"] < 0)
    assert exposed_far.any()
    assert np.all(info["far_ring"][exposed_far])
    assert np.all(info["labels"][exposed_far] == LAYER_FAR_RING)
    assert not np.any(info["near_ring"] & (info["t_ring"] < 0))


def test_transparent_overlap_and_target_occlusion_are_excluded():
    model = setup(spin=1.2)
    x, y = detector_xy_grids(48, 64)
    source = FramePose(1., .2, 32., 24.)
    reference = FramePose(0., 0., 32., 24.)
    info = model.classify_detector(x, y, source)
    overlap = info["near_ring"] & info["on_globe"]
    xd, yd, valid = model.src_to_ref(x, y, source, reference)
    assert overlap.any() and not valid[overlap].any()
    gx, gy, gv = model.globe_model.src_to_ref(x, y, source, reference)
    target = model.classify_detector(gx, gy, reference)
    occluded = (info["labels"] == LAYER_GLOBE) & gv & target["near_ring"] & target["on_globe"]
    assert occluded.any() and not valid[occluded].any()


@pytest.mark.parametrize("mode", ["mono", "RGB", "RGGB"])
def test_stack_does_not_splat_between_layers_or_shadow_regions(mode):
    model = setup()
    x, y = detector_xy_grids(48, 64)
    times = np.array([0., .5, 1.])
    radiances = np.array([1., 10., 30., 3., 7.])
    frames = []
    labels = cfa_labels(48, 64, "RGGB")
    for t in times:
        pose = FramePose(t, .3*t, 32., 24.)
        regions = model.reconstruction_regions(model.classify_detector(x, y, pose))
        frame = np.where(regions >= 0, radiances[np.maximum(regions, 0)], 900.)
        if mode == "RGB":
            frame = frame[..., None] * np.array([1., 2., 3.])
        elif mode == "RGGB":
            frame = frame * np.where(labels == "R", 1., np.where(labels == "G", 2., 3.))
        frames.append(frame)
    src = ArraySource(np.stack(frames), bit_depth=32, color_mode=mode, timestamps=times)
    result = stack_source(src, config())
    target = model.reconstruction_regions(model.classify_detector(x, y, FramePose(0., 0., 32., 24.)))
    expected = radiances[np.maximum(target, 0)]
    if mode != "mono":
        expected = expected[..., None] * np.array([1., 2., 3.])
    np.testing.assert_allclose(result.image[result.validity], expected[result.validity], atol=1e-10)
    assert not np.any(result.validity[target < 0])
    gcov, rcov = result.layer_coverage["globe"], result.layer_coverage["ring"]
    assert gcov.max() > 0 and rcov.max() > 0
    assert not np.any((gcov > 0) & (rcov > 0))
    assert not np.any(gcov[~np.isin(target, [1, 3])])
    assert not np.any(rcov[~np.isin(target, [2, 4])])


def test_edge_on_band_is_actually_masked_including_globe_overlap():
    model = setup(opening=0.)
    x, y = detector_xy_grids(48, 64)
    pose = FramePose(0., 0., 32., 24.)
    info = model.classify_detector(x, y, pose)
    band = info["ring_degenerate"]
    assert band.any() and np.any(band & info["on_globe"])
    _, _, valid = model.src_to_ref(x, y, pose, pose)
    assert not valid[band].any()
    src = ArraySource(np.where(band, 900., 10.)[None], bit_depth=32, timestamps=np.array([0.]))
    with pytest.raises(ValueError, match="edge-on rings cannot be reconstructed"):
        stack_source(src, config(sub_obs_lat_rad=0.))


def test_moon_track_uses_reference_detector_axes_and_nonzero_field_angle():
    moon = MoonTrack(46., 25., 2., 0., 3.)
    angle = .6
    x, y = detector_xy_grids(48, 64)
    for t in (0., 1.):
        pose = FramePose(t, angle, 32., 24.)
        mask = moon_mask(x, y, pose, moon, 0., angle)
        expected = np.hypot(x - 46., y - (25. + 3*t)) <= 2.
        np.testing.assert_array_equal(mask, expected)
        base = setup()
        model = SaturnSceneModel(base.globe, base.rings, moon=moon, field_angle0_rad=angle)
        labels = model.classify_detector(x, y, pose)["labels"]
        np.testing.assert_array_equal(labels == LAYER_MOON, expected)
        img = render_saturn(48, 64, pose, base.globe, base.rings, lambda lon, lat: 0.,
                            ring_tex=lambda r, a: 0., moon=moon, field_angle0_rad=angle)
        np.testing.assert_array_equal(img == 1.35, expected)


def test_disabled_field_rotation_applies_to_rings_and_background():
    base = setup(spin=0.)
    model = SaturnSceneModel(base.globe, base.rings, apply_field=False, apply_surface=False)
    x, y = detector_xy_grids(48, 64)
    dx, dy, valid = model.src_to_ref(x, y, FramePose(0., 1., 32., 24.), FramePose(0., 0., 32., 24.))
    np.testing.assert_allclose(dx[valid], x[valid], atol=1e-10)
    np.testing.assert_allclose(dy[valid], y[valid], atol=1e-10)


def test_moving_moon_requires_known_cadence():
    src = ArraySource(np.ones((2, 48, 64)))
    cfg = config(field_rate_rad_s=0., surface_rate_rad_s=0., moon_x=46., moon_y=24.,
                 moon_radius_px=2., moon_vx_px_s=3.)
    with pytest.raises(ValueError, match="cadence_s"):
        prepare_geometry(src, cfg)


@pytest.mark.parametrize("overrides", [dict(sub_obs_lat_rad=None), dict(equatorial_radius_px=None),
                                        dict(ring_inner_radius_px=None), dict(ring_outer_radius_px=None)])
def test_saturn_requires_declared_physical_geometry(overrides):
    src = ArraySource(np.ones((1, 48, 64)), timestamps=np.array([0.]))
    reason = 'viewing latitude unavailable' if 'sub_obs_lat_rad' in overrides else 'requires'
    with pytest.raises(ValueError, match=reason):
        prepare_geometry(src, config(**overrides))


def test_saturn_exposure_warning_uses_outer_ring_motion():
    src = ArraySource(np.ones((1, 48, 64)), timestamps=np.array([0.]))
    _, _, _, warnings = prepare_geometry(src, config(field_rate_rad_s=.1, surface_rate_rad_s=0., exposure_s=.2))
    assert any("exposure_geometry_frozen" in warning for warning in warnings)


@pytest.mark.parametrize("kwargs", [dict(moon_x=1., moon_y=2., moon_radius_px=1.),
    dict(moon_vx_px_s=1.), dict(sun_lat_rad=.1), dict(ring_transmission=.2)])
def test_saturn_settings_cannot_be_silently_ignored(kwargs):
    with pytest.raises(ValueError, match="saturn|moon track"):
        ReconstructionConfig(**kwargs)


def test_rings_must_not_intersect_globe_and_opaque_means_zero_transmission():
    with pytest.raises(ValueError, match="outside"):
        SaturnSceneModel(GlobeParams(12.), RingParams(10., 20.))
    globe = GlobeParams(12., sub_obs_lat_rad=np.pi/2)
    assert ring_transmission(globe, RingParams(16., 26., transmission=0.)) == 0.


def test_layers_survive_worker_preview_checkpoint_and_cli(tmp_path):
    from planetrecon.jobs import start_stack_job, load_checkpoint
    from planetrecon.cli import main
    model = setup()
    frame = render_saturn(48, 64, FramePose(0., 0., 32., 24.), model.globe, model.rings,
                          lambda lon, lat: 50., ring_tex=lambda r, a: 100.)
    path = tmp_path / 'saturn.ser'
    write_ser(path, np.stack([frame.astype(np.uint8)] * 2))
    cfg = config(field_rate_rad_s=0., surface_rate_rad_s=0., cadence_s=.1)
    job = start_stack_job(path, cfg, job_id='saturn-review', checkpoint_dir=tmp_path, queue_size=16)
    events = []
    try:
        deadline = time.monotonic() + 20.
        while job.state == 'running' and time.monotonic() < deadline:
            events.extend(job.poll(timeout=.1))
        assert job.state == 'completed', [(e.kind, e.payload.get('message')) for e in events]
        completed = next(e.payload for e in events if e.kind == 'completed')
        assert set(completed['layer_coverage']) == {'globe', 'ring'}
        previews = [e.payload for e in events if e.kind == 'preview']
        assert previews and set(previews[-1]['layer_coverage']) == {'globe', 'ring'}
        loaded = load_checkpoint(tmp_path / 'saturn-review.npz', cfg)
        for name in ('globe', 'ring'):
            np.testing.assert_array_equal(loaded['layer_coverage'][name], completed['layer_coverage'][name])
    finally:
        job.close()
    out = tmp_path / 'cli'
    assert main(['--threads', '2', 'stack', '--path', str(path), '--out', str(out), '--device', 'cpu',
        '--geometry', 'saturn', '--radius', '12', '--flattening', '.1', '--sub-obs-lat-deg', str(np.degrees(.4)),
        '--ring-inner', '16', '--ring-outer', '26', '--cadence', '.1', '--field-rate-deg-s', '0',
        '--surface-rate-deg-s', '0', '--no-frame-preselection']) == 0
    with np.load(out / 'stack.npz', allow_pickle=False) as data:
        for name in ('globe', 'ring'):
            assert data[f'layer_coverage__{name}'].shape == (48, 64)
            assert np.max(data[f'layer_coverage__{name}']) > 0
        assert json.loads(str(data['provenance']))['geometry_operator_version'] == cfg.geometry_operator_version


def test_fully_masked_capture_is_not_reported_as_success():
    source = ArraySource(np.full((1, 48, 64), 100.), bit_depth=32, timestamps=np.array([0.]))
    with pytest.raises(ValueError, match="no usable frames"):
        stack_source(source, config(moon_x=32., moon_y=24., moon_radius_px=100.))


def test_masked_moon_brightness_does_not_change_quality_weights():
    moon = MoonTrack(34., 24., 2.)
    model = setup(moon=moon)
    x, y = detector_xy_grids(48, 64)
    pose = FramePose(0., 0., 32., 24.)
    mask = moon_mask(x, y, pose, moon, 0.)
    texture = 30. + np.sin(x/2) * np.cos(y/3)
    cfg = config(field_rate_rad_s=0., surface_rate_rad_s=0., moon_x=34., moon_y=24., moon_radius_px=2.)
    results = [stack_source(ArraySource(np.where(mask, brightness, texture)[None], bit_depth=32,
                                       timestamps=np.array([0.])), cfg) for brightness in (100., 10000.)]
    np.testing.assert_allclose(results[0].coverage, results[1].coverage, atol=1e-12)
    np.testing.assert_allclose(results[0].image, results[1].image, atol=1e-12)


@pytest.mark.parametrize("field_rate", [0., .3])
def test_field_angle_estimate_uses_static_rings_instead_of_spinning_globe(field_rate):
    model = setup(spin=1.2)
    times = np.arange(5.) * .4
    def bright_spot(lon, lat):
        da = np.arctan2(np.sin(lon-.5), np.cos(lon-.5))
        return 1. + 20.*np.exp(-((da/.2)**2 + (lat/.2)**2)/2)
    # Ring asymmetry is required to distinguish half-turns without a rate prior.
    frames = [render_saturn(48, 64, FramePose(t, field_rate*t, 32., 24.), model.globe, model.rings, bright_spot,
                           ring_tex=lambda r, theta: 1+.4*np.cos(theta))
              for t in times]
    src = ArraySource(np.stack(frames), bit_depth=32, timestamps=times)
    _, _, diag, _ = prepare_geometry(src, config(field_rate_rad_s=None, surface_rate_rad_s=1.2))
    assert diag['field_origin'] == 'inferred'
    assert diag['field_rate_rad_s'] == pytest.approx(field_rate, abs=.04)


def test_saturn_setup_reports_all_missing_values_and_accepts_signed_zero():
    cfg = ReconstructionConfig(geometry_mode='saturn')
    assert set(cfg.missing_saturn_geometry()) == {
        'sub_obs_lat_rad', 'equatorial_radius_px', 'ring_inner_radius_px', 'ring_outer_radius_px'}
    with pytest.raises(ValueError, match='Signed observer latitude.*Globe equatorial radius.*Inner ring radius.*Outer ring radius'):
        cfg.require_saturn_geometry()
    config(sub_obs_lat_rad=0.).require_saturn_geometry()
    config(sub_obs_lat_rad=-.4).require_saturn_geometry()
    ReconstructionConfig().require_saturn_geometry()
