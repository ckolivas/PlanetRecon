import numpy as np
import pytest

from planetrecon.detector import cfa_labels, channel_mask
from planetrecon.geometry.coords import detector_xy_grids
from planetrecon.geometry.fit import estimate_field_angle, sequence_degeneracy
from planetrecon.geometry.globe import GlobeParams, body_to_sky, render_globe_texture, sky_to_body
from planetrecon.geometry.model import FieldOnlyModel, OblateGlobeModel, warp_adjoint_error
from planetrecon.geometry.pose import FramePose, unwrap_angles
from planetrecon.geometry.warp import bilinear_push, bilinear_sample, bilinear_sample_adjoint
from planetrecon.io.source import ArraySource
from planetrecon.operators import adjoint_relative_error
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.geometry_stack import freeze_midexposure_error, prepare_geometry
from planetrecon.reconstruction import ReconstructionConfig


def _spot(y0, x0, h=48, w=48, sy=3.0, sx=4.0):
    yy, xx = np.indices((h, w), dtype=np.float64)
    return np.exp(-0.5 * (((yy - y0) / sy) ** 2 + ((xx - x0) / sx) ** 2))


def _belt_tex(lon, lat):
    return 0.55 + 0.3 * np.cos(4.0 * lat) + 0.2 * np.cos(lon) * np.cos(lat)


def _spot_tex(lon, lat, lon0=0.35, lat0=0.15):
    dlon = np.arctan2(np.sin(lon - lon0), np.cos(lon - lon0))
    return _belt_tex(lon, lat) + 0.9 * np.exp(-0.5 * ((dlon / 0.22) ** 2 + ((lat - lat0) / 0.16) ** 2))


def test_bilinear_sample_adjoint():
    rng = np.random.default_rng(7)
    src = rng.normal(size=(17, 19))
    dest_shape = (13, 15)
    y = rng.uniform(-1.0, 17.0, size=dest_shape)
    x = rng.uniform(-1.0, 19.0, size=dest_shape)
    err = adjoint_relative_error(
        lambda a: bilinear_sample(a, y, x, fill=0.0),
        lambda b: bilinear_sample_adjoint(b, y, x, src.shape),
        src,
        rng.normal(size=dest_shape),
    )
    assert err < 1e-12


def test_bilinear_push_identity_conserves_mass():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(11, 11))
    yy, xx = np.indices(src.shape, dtype=np.float64)
    pushed, weight = bilinear_push(src, yy, xx, src.shape)
    assert pushed.sum() == pytest.approx(src.sum())
    assert weight.sum() == pytest.approx(float(src.size))


def test_field_rotation_identity_and_adjoint():
    h = w = 32
    x, y = detector_xy_grids(h, w)
    pose = FramePose(t_s=0.0, field_angle_rad=0.4, cx=16.0, cy=16.0)
    ref = FramePose(t_s=0.0, field_angle_rad=0.4, cx=16.0, cy=16.0)
    model = FieldOnlyModel()
    xd, yd, valid = model.src_to_ref(x, y, pose, ref)
    assert np.nanmax(np.abs(xd[valid] - x[valid])) < 1e-10
    rng = np.random.default_rng(11)
    assert warp_adjoint_error(model, pose, ref, (h, w), rng) < 1e-11
    src = FramePose(t_s=0.2, field_angle_rad=0.4, cx=16.0, cy=16.0)
    ref0 = FramePose(t_s=0.0, field_angle_rad=0.0, cx=16.0, cy=16.0)
    assert warp_adjoint_error(model, src, ref0, (h, w), rng) < 1e-11


def test_unwrap_crosses_two_pi():
    raw = np.array([3.0, 3.1, -3.1, -2.9])
    out = unwrap_angles(raw)
    assert out[2] > out[1]
    assert np.all(np.diff(out) > -0.5)


def test_globe_centre_pole_and_east_limb():
    globe = GlobeParams(equatorial_radius_px=20.0, flattening=0.0)
    lon, lat, vis, mu = sky_to_body(0.0, 0.0, globe, 0.0)
    assert vis
    assert lon == pytest.approx(0.0, abs=1e-10)
    assert lat == pytest.approx(0.0, abs=1e-10)
    assert mu == pytest.approx(1.0, abs=1e-6)
    sx, sy, vis_n, _ = body_to_sky(0.0, 0.5 * np.pi, globe, 0.0)
    assert vis_n
    assert sx == pytest.approx(0.0, abs=1e-8)
    assert sy == pytest.approx(20.0, abs=1e-8)
    sx_e, sy_e, vis_e, _ = body_to_sky(0.5 * np.pi, 0.0, globe, 0.0)
    assert vis_e
    assert sx_e == pytest.approx(20.0, abs=1e-8)
    assert sy_e == pytest.approx(0.0, abs=1e-8)
    _lon, _lat, vis_far, _ = sky_to_body(0.0, 0.0, globe, 0.0)
    # A sky point well outside the disc is invisible.
    _lon, _lat, vis_out, _ = sky_to_body(40.0, 0.0, globe, 0.0)
    assert not vis_out


def test_oblate_outline_is_ellipse():
    globe = GlobeParams(equatorial_radius_px=20.0, flattening=0.2)
    assert globe.polar_radius_px == pytest.approx(16.0)
    _lon, _lat, vis_eq, _ = sky_to_body(19.5, 0.0, globe, 0.0)
    _lon, _lat, vis_pol, _ = sky_to_body(0.0, 19.5, globe, 0.0)
    assert vis_eq
    assert not vis_pol
    _lon, _lat, vis_pol_in, _ = sky_to_body(0.0, 15.5, globe, 0.0)
    assert vis_pol_in


def test_globe_render_adjoint_via_composed_warp():
    globe = GlobeParams(equatorial_radius_px=14.0, flattening=0.06, surface_rate_rad_s=0.4)
    model = OblateGlobeModel(globe, apply_field=True, apply_surface=True)
    src = FramePose(t_s=0.5, field_angle_rad=0.2, cx=18.0, cy=18.0)
    ref = FramePose(t_s=0.0, field_angle_rad=0.0, cx=18.0, cy=18.0)
    rng = np.random.default_rng(5)
    assert warp_adjoint_error(model, src, ref, (36, 36), rng) < 1e-11


def test_zero_motion_matches_mean_stack():
    frames = np.stack([_spot(20, 22) + 0.2 * _spot(30, 12, sy=5, sx=3) for _ in range(4)])
    src = ArraySource(frames, color_mode="mono", bit_depth=32, timestamps=np.arange(4, dtype=float))
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="field",
        field_rate_rad_s=0.0,
        field_center_x=24.0,
        field_center_y=24.0,
        equatorial_radius_px=20.0,
        freeze_mid_exposure=False,
    )
    result = stack_source(src, cfg)
    assert result.n_used == 4
    interior = result.validity[8:40, 8:40]
    err = np.linalg.norm((result.image - frames[0])[8:40, 8:40][interior]) / np.linalg.norm(frames[0][8:40, 8:40])
    assert err < 1e-6


def test_field_only_recovers_common_view():
    h = w = 48
    truth = _spot(18, 30, h, w) + 0.5 * _spot(32, 16, h, w, sy=4, sx=2.5)
    times = np.array([0.0, 0.2, 0.4, 0.6])
    rate = 0.7
    cx = cy = 24.0
    model = FieldOnlyModel()
    ref = FramePose(t_s=0.0, field_angle_rad=0.0, cx=cx, cy=cy)
    frames = []
    for t in times:
        src = FramePose(t_s=float(t), field_angle_rad=rate * float(t), cx=cx, cy=cy)
        from planetrecon.geometry.model import render_observed

        frames.append(render_observed(truth, model, src, ref))
    src = ArraySource(np.stack(frames), color_mode="mono", bit_depth=32, timestamps=times)
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="field",
        field_rate_rad_s=rate,
        field_center_x=cx,
        field_center_y=cy,
        freeze_mid_exposure=False,
    )
    result = stack_source(src, cfg)
    mask = (result.coverage > 0.05 * float(np.max(result.coverage))) & (truth > 0.05)
    corr = np.corrcoef(result.image[mask], truth[mask])[0, 1]
    assert corr > 0.98
    assert result.provenance["geometry"]["field_origin"] == "user"


def test_spin_only_does_not_fill_unseen_longitudes():
    h = w = 48
    cx = cy = 24.0
    radius = 16.0
    globe = GlobeParams(
        equatorial_radius_px=radius,
        flattening=0.05,
        surface_rate_rad_s=0.9,
        reference_epoch_s=0.0,
    )
    times = np.array([0.0, 0.4, 0.8])
    frames = []
    for t in times:
        frames.append(
            render_globe_texture(h, w, cx, cy, 0.0, globe, float(t), _spot_tex, apply_field=False)
        )
    src = ArraySource(np.stack(frames), color_mode="mono", bit_depth=32, timestamps=times)
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="surface",
        surface_rate_rad_s=0.9,
        field_center_x=cx,
        field_center_y=cy,
        equatorial_radius_px=radius,
        flattening=0.05,
        freeze_mid_exposure=False,
    )
    result = stack_source(src, cfg)
    truth = render_globe_texture(h, w, cx, cy, 0.0, globe, 0.0, _spot_tex, apply_field=False)
    on_disc = truth > 1e-6
    yy, xx = np.indices((h, w), dtype=np.float64)
    far = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) > radius + 2.5
    assert float(np.max(np.abs(result.image[far]))) < 0.05
    mask = (result.coverage > 0.05 * float(np.max(result.coverage))) & on_disc
    corr = np.corrcoef(result.image[mask], truth[mask])[0, 1]
    assert corr > 0.97
    assert np.any(result.coverage[on_disc] > 0)


def test_combined_field_and_surface():
    h = w = 48
    cx = cy = 24.0
    radius = 16.0
    globe = GlobeParams(
        equatorial_radius_px=radius,
        flattening=0.04,
        surface_rate_rad_s=0.5,
        reference_epoch_s=0.0,
    )
    times = np.array([0.0, 0.3, 0.6])
    field_rate = 0.45
    frames = []
    for t in times:
        frames.append(
            render_globe_texture(
                h, w, cx, cy, field_rate * float(t), globe, float(t), _spot_tex, apply_field=True
            )
        )
    src = ArraySource(np.stack(frames), color_mode="mono", bit_depth=32, timestamps=times)
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="combined",
        field_rate_rad_s=field_rate,
        surface_rate_rad_s=0.5,
        field_center_x=cx,
        field_center_y=cy,
        equatorial_radius_px=radius,
        flattening=0.04,
        freeze_mid_exposure=False,
    )
    result = stack_source(src, cfg)
    truth = render_globe_texture(h, w, cx, cy, 0.0, globe, 0.0, _spot_tex, apply_field=True)
    mask = (result.coverage > 0.05 * float(np.max(result.coverage))) & (truth > 1e-4)
    corr = np.corrcoef(result.image[mask], truth[mask])[0, 1]
    assert corr > 0.95


def test_featureless_disc_reports_roll_degeneracy():
    yy, xx = np.indices((40, 40), dtype=np.float64)
    r = np.hypot(xx - 20.0, yy - 20.0)
    disc = np.clip(1.0 - (r / 12.0) ** 2, 0.0, None)
    frames = np.stack([disc, disc, disc])
    flags = sequence_degeneracy(list(frames))
    assert "roll_unconstrained" in flags
    src = ArraySource(frames, color_mode="mono", bit_depth=32, timestamps=np.arange(3, dtype=float))
    cfg = ReconstructionConfig(device="cpu", threads=2, geometry_mode="field", freeze_mid_exposure=False)
    result = stack_source(src, cfg)
    assert any("roll_unconstrained" in w for w in result.warnings)


def test_angle_wrap_estimation():
    h = w = 48
    truth = _spot(16, 32, h, w) + _spot(30, 18, h, w, sy=2.5, sx=3.5)
    cx = cy = 24.0
    model = FieldOnlyModel()
    ref = FramePose(t_s=0.0, field_angle_rad=0.0, cx=cx, cy=cy)
    from planetrecon.geometry.model import render_observed

    angles = np.array([3.0, 3.2, -3.05])
    frames = []
    for ang in angles:
        frames.append(render_observed(truth, model, FramePose(0.0, float(ang), cx, cy), ref))
    unwrapped = unwrap_angles(angles)
    assert unwrapped[2] > unwrapped[1]
    est = estimate_field_angle(frames[0], frames[1], cx, cy, 18.0)
    dtheta = (3.2 - 3.0)
    err = abs((est["angle_rad"] - dtheta + np.pi) % (2 * np.pi) - np.pi)
    assert err < 0.12, est


def test_cfa_stays_on_detector_lattice():
    h = w = 32
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    rgb[..., 0] = _spot(10, 20, h, w)
    rgb[..., 1] = _spot(16, 16, h, w, sy=6, sx=6)
    rgb[..., 2] = _spot(22, 12, h, w)
    rate = 0.5
    times = np.array([0.0, 0.4])
    model = FieldOnlyModel()
    ref = FramePose(0.0, 0.0, 16.0, 16.0)
    from planetrecon.geometry.model import render_observed

    raw_frames = []
    for t in times:
        rotated = render_observed(rgb, model, FramePose(float(t), rate * float(t), 16.0, 16.0), ref)
        labels = cfa_labels(h, w, "RGGB")
        raw = np.zeros((h, w), dtype=np.float64)
        raw[channel_mask(labels, "R")] = rotated[..., 0][channel_mask(labels, "R")]
        raw[channel_mask(labels, "G")] = rotated[..., 1][channel_mask(labels, "G")]
        raw[channel_mask(labels, "B")] = rotated[..., 2][channel_mask(labels, "B")]
        raw_frames.append(raw)
    src = ArraySource(np.stack(raw_frames), color_mode="RGGB", bit_depth=32, timestamps=times)
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="field",
        field_rate_rad_s=rate,
        field_center_x=16.0,
        field_center_y=16.0,
        freeze_mid_exposure=False,
        reject_saturated=False,
    )
    result = stack_source(src, cfg)
    assert result.image.ndim == 3
    # Detector-fixed CFA: each colour plane is supported only where samples landed.
    assert np.any(result.coverage[..., 0] > 0) and np.any(result.coverage[..., 0] == 0)
    assert result.provenance["demosaic_first"]["label"] == "comparison"


def test_long_clip_and_exposure_warnings():
    frames = np.stack([_spot(16, 16, 32, 32) for _ in range(3)])
    src = ArraySource(frames, color_mode="mono", bit_depth=32, timestamps=np.array([0.0, 80.0, 160.0]))
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="field",
        field_rate_rad_s=0.02,
        field_center_x=16.0,
        field_center_y=16.0,
        equatorial_radius_px=12.0,
        exposure_s=2.0,
        freeze_mid_exposure=True,
        geometry_duration_warn_s=100.0,
    )
    result = stack_source(src, cfg)
    text = " ".join(result.warnings)
    assert "rigid_rotation_duration_limit" in text
    assert "exposure_geometry_frozen" in text


def test_freeze_midexposure_error_grows_with_rate():
    h = w = 36
    truth = _spot(18, 18, h, w, sy=6, sx=6)
    model = FieldOnlyModel()
    ref = FramePose(0.0, 0.0, 18.0, 18.0)

    def make_pose(t, rate):
        return FramePose(float(t), float(rate) * float(t), 18.0, 18.0)

    slow = freeze_midexposure_error(truth, model, 0.0, 0.2, ref, lambda t: make_pose(t, 0.05), n_quad=5)
    fast = freeze_midexposure_error(truth, model, 0.0, 0.2, ref, lambda t: make_pose(t, 2.5), n_quad=5)
    assert fast > slow


def test_config_rejects_unknown_geometry_mode():
    with pytest.raises(ValueError, match="geometry_mode"):
        ReconstructionConfig(geometry_mode="saturn")
    with pytest.raises(ValueError, match="geometry operator"):
        ReconstructionConfig(geometry_operator_version="0.0")


def test_prepare_geometry_uses_user_rates():
    frames = np.stack([_spot(12, 12, 24, 24) for _ in range(3)])
    src = ArraySource(frames, color_mode="mono", bit_depth=32, timestamps=np.arange(3, dtype=float) * 0.1)
    cfg = ReconstructionConfig(
        device="cpu",
        threads=2,
        geometry_mode="combined",
        field_rate_rad_s=0.1,
        surface_rate_rad_s=0.2,
        field_center_x=12.0,
        field_center_y=12.0,
        equatorial_radius_px=8.0,
        freeze_mid_exposure=False,
    )
    poses, model, diag, warnings = prepare_geometry(src, cfg)
    assert isinstance(model, OblateGlobeModel)
    assert diag["field_origin"] == "user"
    assert diag["surface_origin"] == "user"
    assert poses[1].field_angle_rad == pytest.approx(0.01)
