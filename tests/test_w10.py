import numpy as np
import pytest

from planetrecon.geometry.coords import detector_to_sky, detector_xy_grids
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import warp_adjoint_error
from planetrecon.geometry.pose import FramePose
from planetrecon.geometry.rings import RingParams, intersect_ring
from planetrecon.geometry.saturn import (
    LAYER_FAR_RING,
    LAYER_GLOBE,
    LAYER_MOON,
    LAYER_NEAR_RING,
    MoonTrack,
    SaturnSceneModel,
    classify_layers,
    fit_saturn_geometry,
    render_saturn,
)
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def _globe_tex(lon, lat):
    dlon = np.arctan2(np.sin(lon - 0.4), np.cos(lon - 0.4))
    return 0.5 + 0.2 * np.cos(3 * lat) + 0.7 * np.exp(-0.5 * ((dlon / 0.25) ** 2 + ((lat - 0.1) / 0.18) ** 2))


def _ring_clump(radius, azimuth, az0=0.7):
    x = (np.asarray(radius) - 16.0) / 10.0
    base = 0.3 + 0.7 * np.exp(-((x - 0.4) / 0.25) ** 2)
    daz = np.arctan2(np.sin(azimuth - az0), np.cos(azimuth - az0))
    return base + 0.9 * np.exp(-0.5 * (daz / 0.2) ** 2)


def _setup(opening=0.4, spin=0.0, sun_lon=None, sun_lat=None):
    globe = GlobeParams(
        equatorial_radius_px=12.0,
        flattening=0.1,
        sub_obs_lat_rad=opening,
        surface_rate_rad_s=spin,
        reference_epoch_s=0.0,
    )
    rings = RingParams(16.0, 26.0, transmission=0.4, sun_lon_rad=sun_lon, sun_lat_rad=sun_lat)
    return globe, rings


def test_open_rings_have_ansae_and_near_front():
    globe, rings = _setup(0.4)
    x, y = detector_xy_grids(48, 48)
    sx, sy = detector_to_sky(x, y, 24.0, 24.0)
    info = classify_layers(sx, sy, globe, rings, 0.0)
    assert np.any(info["near_ring"]) and np.any(info["far_ring"])
    assert np.any(info["on_globe"])
    overlap_near = info["near_ring"] & info["on_globe"]
    assert np.any(overlap_near)
    # Near ring is in front: those overlap pixels are classified as near ring, not globe.
    assert np.all(info["labels"][overlap_near] == LAYER_NEAR_RING)
    far_behind = info["on_globe"] & info["far_ring"]
    if np.any(far_behind):
        assert np.all(info["labels"][far_behind] == LAYER_GLOBE)


def test_both_opening_signs_flip_near_far_z():
    x, y = detector_xy_grids(48, 48)
    sx, sy = detector_to_sky(x, y, 24.0, 24.0)
    globe_p, rings = _setup(0.4)
    globe_n, _ = _setup(-0.4)
    ip = classify_layers(sx, sy, globe_p, rings, 0.0)
    inn = classify_layers(sx, sy, globe_n, rings, 0.0)
    assert ip["near_ring"].sum() > 10 and inn["near_ring"].sum() > 10
    # Opposite opening puts the same sky pixel on opposite ring sides.
    both = ip["on_ring"] & inn["on_ring"]
    assert np.any(ip["near_ring"][both] != inn["near_ring"][both])


def test_edge_on_rings_are_degenerate():
    globe, rings = _setup(0.0)
    _r, _a, _t, on, edge = intersect_ring(0.0, 0.0, globe, rings)
    assert edge
    assert not bool(on)
    model = SaturnSceneModel(globe, rings)
    assert model.edge_on
    pose = FramePose(0.0, 0.0, 24.0, 24.0)
    x, y = detector_xy_grids(32, 32)
    _dx, _dy, valid = model.src_to_ref(x, y, pose, pose)
    info = model.classify_detector(x, y, pose)
    ring = (info["labels"] == LAYER_NEAR_RING) | (info["labels"] == LAYER_FAR_RING)
    assert not np.any(valid & ring)


def test_globe_spin_does_not_move_rings():
    globe, rings = _setup(0.4, spin=0.8)
    pose0 = FramePose(0.0, 0.0, 24.0, 24.0)
    pose1 = FramePose(0.8, 0.0, 24.0, 24.0)
    img0 = render_saturn(48, 48, pose0, globe, rings, _globe_tex, ring_tex=_ring_clump)
    img1 = render_saturn(48, 48, pose1, globe, rings, _globe_tex, ring_tex=_ring_clump)
    x, y = detector_xy_grids(48, 48)
    sx, sy = detector_to_sky(x, y, 24.0, 24.0)
    info = classify_layers(sx, sy, globe, rings, 0.0)
    ring = info["on_ring"] & ~info["on_globe"]
    globe_only = info["on_globe"] & ~info["on_ring"]
    # Ring clump is inertial; globe spot moves.
    assert np.corrcoef(img0[ring], img1[ring])[0, 1] > 0.98
    assert np.corrcoef(img0[globe_only], img1[globe_only])[0, 1] < 0.97


def test_saturn_stack_recovers_layers():
    globe, rings = _setup(0.4, spin=0.5)
    times = np.array([0.0, 0.4, 0.8])
    frames = [
        render_saturn(48, 48, FramePose(float(t), 0.0, 24.0, 24.0), globe, rings, _globe_tex, ring_tex=_ring_clump)
        for t in times
    ]
    src = ArraySource(np.stack(frames), color_mode="mono", bit_depth=32, timestamps=times)
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="saturn",
        surface_rate_rad_s=0.5,
        field_rate_rad_s=0.0,
        field_center_x=24.0,
        field_center_y=24.0,
        equatorial_radius_px=12.0,
        flattening=0.1,
        sub_obs_lat_rad=0.4,
        ring_inner_radius_px=16.0,
        ring_outer_radius_px=26.0,
        freeze_mid_exposure=False,
    )
    result = stack_source(src, cfg)
    truth = frames[0]
    cov = result.coverage if result.coverage.ndim == 2 else result.coverage[..., 0]
    mask = cov > 0.05 * float(np.max(cov))
    corr = np.corrcoef(result.image[mask], truth[mask])[0, 1]
    assert corr > 0.9
    assert "globe" in result.layer_coverage and "ring" in result.layer_coverage
    assert float(np.max(result.layer_coverage["globe"])) > 0
    assert float(np.max(result.layer_coverage["ring"])) > 0
    # Globe and ring coverage are not the same map.
    g = result.layer_coverage["globe"] > 0
    r = result.layer_coverage["ring"] > 0
    assert np.any(g & ~r) and np.any(r)


def test_field_rotation_moves_entire_composite():
    globe, rings = _setup(0.35, spin=0.0)
    a = render_saturn(48, 48, FramePose(0.0, 0.0, 24.0, 24.0), globe, rings, _globe_tex)
    b = render_saturn(48, 48, FramePose(0.0, 0.4, 24.0, 24.0), globe, rings, _globe_tex)
    assert np.linalg.norm(a - b) / np.linalg.norm(a) > 0.05


def test_globe_shadow_on_far_ring():
    globe, rings = _setup(0.4, sun_lon=0.0, sun_lat=0.35)
    x, y = detector_xy_grids(48, 48)
    sx, sy = detector_to_sky(x, y, 24.0, 24.0)
    info = classify_layers(sx, sy, globe, rings, 0.0)
    assert np.any(info["globe_shadow"] & info["far_ring"])
    lit = render_saturn(48, 48, FramePose(0.0, 0.0, 24.0, 24.0), globe, rings, _globe_tex)
    dark_g, dark_r = _setup(0.4, sun_lon=np.pi, sun_lat=0.2)
    dark = render_saturn(48, 48, FramePose(0.0, 0.0, 24.0, 24.0), dark_g, dark_r, _globe_tex)
    # A sun on the far side darkens more of the visible globe.
    assert dark[info["on_globe"]].mean() < lit[info["on_globe"]].mean()


def test_moving_moon_is_masked_not_baked_into_globe():
    globe, rings = _setup(0.4, spin=0.0)
    moon = MoonTrack(x=36.0, y=24.0, radius_px=2.5, vx_px_s=6.0, vy_px_s=0.0)
    times = np.array([0.0, 0.5])
    frames = [
        render_saturn(
            48, 48, FramePose(float(t), 0.0, 24.0, 24.0), globe, rings, _globe_tex, moon=moon
        )
        for t in times
    ]
    assert frames[0][24, 36] > 1.2
    assert frames[1][24, 36] < 1.2
    src = ArraySource(np.stack(frames), color_mode="mono", bit_depth=32, timestamps=times)
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="saturn",
        surface_rate_rad_s=0.0,
        field_rate_rad_s=0.0,
        field_center_x=24.0,
        field_center_y=24.0,
        equatorial_radius_px=12.0,
        flattening=0.1,
        sub_obs_lat_rad=0.4,
        ring_inner_radius_px=16.0,
        ring_outer_radius_px=26.0,
        moon_x=36.0,
        moon_y=24.0,
        moon_radius_px=2.5,
        moon_vx_px_s=6.0,
        freeze_mid_exposure=False,
    )
    result = stack_source(src, cfg)
    x, y = detector_xy_grids(48, 48)
    model = SaturnSceneModel(globe, rings, moon=moon)
    labels = model.classify_detector(x, y, FramePose(0.0, 0.0, 24.0, 24.0))["labels"]
    assert np.any(labels == LAYER_MOON)
    # The moon track is excluded from the globe/ring canvases.
    assert result.image[24, 36] < 1.2


def test_ansae_do_not_set_globe_radius():
    globe, rings = _setup(0.45)
    img = render_saturn(48, 48, FramePose(0.0, 0.0, 24.0, 24.0), globe, rings, _globe_tex)
    from planetrecon.geometry.fit import fit_disc_ellipse

    disc = fit_disc_ellipse(img)
    sat = fit_saturn_geometry(img)
    assert sat["ok"]
    assert sat["radius"] < disc["radius"]
    assert sat["radius"] < 16.0
    assert sat["ring_outer"] is not None and sat["ring_outer"] > sat["radius"]


def test_saturn_warp_adjoint():
    globe, rings = _setup(0.35, spin=0.3)
    model = SaturnSceneModel(globe, rings)
    src = FramePose(0.4, 0.15, 24.0, 24.0)
    ref = FramePose(0.0, 0.0, 24.0, 24.0)
    rng = np.random.default_rng(4)
    assert warp_adjoint_error(model, src, ref, (32, 32), rng) < 1e-11


def test_config_requires_saturn_for_rings():
    with pytest.raises(ValueError, match="ring radii"):
        ReconstructionConfig(geometry_mode="field", ring_inner_radius_px=10.0, ring_outer_radius_px=20.0)
    ReconstructionConfig(
        geometry_mode="saturn",
        ring_inner_radius_px=10.0,
        ring_outer_radius_px=20.0,
        equatorial_radius_px=8.0,
    )


def test_low_opening_masks_overlap():
    globe, rings = _setup(0.08)
    model = SaturnSceneModel(globe, rings)
    assert model.low_opening
    pose = FramePose(0.0, 0.0, 24.0, 24.0)
    x, y = detector_xy_grids(48, 48)
    _dx, _dy, valid = model.src_to_ref(x, y, pose, pose)
    info = model.classify_detector(x, y, pose)
    overlap = info["on_globe"] & info["on_ring"]
    assert np.any(overlap)
    assert not np.any(valid[overlap])
