"""Regression cases for the W09 commit review."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.calibration import Calibration
from planetrecon.geometry.fit import fit_disc_ellipse
from planetrecon.geometry.globe import GlobeParams
from planetrecon.geometry.model import FieldOnlyModel, render_observed
from planetrecon.geometry.pose import FramePose, source_times_s
from planetrecon.geometry.warp import bilinear_sample, bilinear_sample_adjoint
from planetrecon.io.ser import SERSource, write_ser
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def texture():
    yy, xx = np.indices((48, 48), dtype=float)
    return (np.exp(-((xx - 31)**2 + (yy - 17)**2) / 12)
            + .6 * np.exp(-((xx - 17)**2 + (yy - 29)**2) / 18))


def config(**kwargs):
    return ReconstructionConfig(device="cpu", threads=2, geometry_mode="field",
                                field_center_x=24., field_center_y=24.,
                                equatorial_radius_px=20., **kwargs)


def rotated_source(times, rate):
    ref = FramePose(0., 0., 24., 24.)
    model = FieldOnlyModel()
    frames = [render_observed(texture(), model, FramePose(t, rate*t, 24., 24.), ref)
              for t in times]
    return ArraySource(np.stack(frames), bit_depth=32, timestamps=np.asarray(times))


def test_inferred_rate_uses_actual_sparse_sample_times():
    times = np.arange(21) * .1
    source = rotated_source(times, .3)
    result = stack_source(source, config(reference_index=3))
    diag = result.provenance["geometry"]
    assert diag["sample_indices"] == [0, 3, 10, 20]
    assert diag["sample_times_s"] == pytest.approx(times[[0, 3, 10, 20]])
    assert diag["field_rate_rad_s"] == pytest.approx(.3, abs=.035)
    mask = texture() > .05
    assert np.corrcoef(result.image[mask], texture()[mask])[0, 1] > .98


def test_inferred_exposure_warning_uses_estimated_rate():
    source = rotated_source(np.arange(3) * .5, .8)
    result = stack_source(source, config(exposure_s=1.))
    assert any("exposure_geometry_frozen" in warning for warning in result.warnings)


def test_reference_epoch_does_not_move_to_exposure_midpoint():
    # Render at the actual exposure midpoint, reconstruct at the declared epoch.
    ref = FramePose(0., 0., 24., 24.)
    frame = render_observed(texture(), FieldOnlyModel(), FramePose(.5, .3, 24., 24.), ref)
    source = ArraySource(frame[None], bit_depth=32, timestamps=np.array([0.]))
    result = stack_source(source, config(field_rate_rad_s=.6, exposure_s=1.))
    yy, xx = np.indices(frame.shape)
    original_cx = np.sum(texture() * xx) / texture().sum()
    output_cx = np.sum(result.image * xx) / result.image.sum()
    assert output_cx == pytest.approx(original_cx, abs=.1)
    assert result.reference_epoch == "0.0"
    assert result.provenance["geometry"]["reference_epoch_s"] == 0.


def test_ser_timestamp_ticks_are_differenced_before_float_conversion(tmp_path):
    ticks = np.array([638_900_000_000_000_000, 638_900_000_000_000_001,
                      638_900_000_000_000_102], dtype=np.uint64)
    path = tmp_path / "ticks.ser"
    write_ser(path, np.zeros((3, 8, 8), dtype=np.uint8), timestamps=ticks)
    with SERSource(path) as source:
        times, origin = source_times_s(source)
    assert origin == "measured"
    np.testing.assert_allclose(times, [0., 1e-7, 1.02e-5], rtol=1e-14)


def test_array_timestamp_units_do_not_depend_on_magnitude():
    src = ArraySource(np.zeros((2, 8, 8)), timestamps=np.array([1e13, 1e13 + 2]))
    times, _ = source_times_s(src)
    np.testing.assert_array_equal(times, [0., 2.])


@pytest.mark.parametrize("times", [[0., 2., 1.], [2., 1., 1.]])
def test_bad_time_order_is_rejected(times):
    src = ArraySource(np.zeros((3, 8, 8)), timestamps=np.array(times))
    with pytest.raises(ValueError, match="nondecreasing"):
        source_times_s(src)


def test_physical_rate_requires_known_time_scale():
    src = ArraySource(np.stack([texture(), texture()]), bit_depth=32)
    with pytest.raises(ValueError, match="cadence_s"):
        stack_source(src, config(field_rate_rad_s=.2))
    result = stack_source(src, config(field_rate_rad_s=.2, cadence_s=.1))
    assert result.provenance["geometry"]["time_origin"] == "user"


def test_bad_sample_frame_does_not_prevent_valid_stack():
    src = ArraySource(np.stack([np.full((48, 48), np.nan), texture(), texture()]), bit_depth=32,
                      timestamps=np.arange(3.))
    cfg = ReconstructionConfig(device="cpu", threads=2, geometry_mode="field", field_rate_rad_s=0.)
    result = stack_source(src, cfg)
    assert (result.n_used, result.n_rejected) == (2, 1)
    assert result.provenance["geometry"]["sample_indices"] == [1, 2]
    assert result.provenance["reference_index"] == 1


def test_pose_fit_uses_calibrated_samples_and_rejects_selected_bad_reference():
    bias = np.full((48, 48), 50.)
    frames = np.stack([bias + texture() for _ in range(3)])
    frames[1] = 100.
    src = ArraySource(frames, bit_depth=32, timestamps=np.arange(3.))
    cal = Calibration(bias=bias, saturate_adu=99.)
    result = stack_source(src, config(field_rate_rad_s=0.), calibration=cal)
    expected = fit_disc_ellipse(texture())
    assert result.provenance["geometry"]["disc"]["cx"] == pytest.approx(expected["cx"])
    assert result.n_rejected == 1
    with pytest.raises(ValueError, match="reference frame"):
        stack_source(src, config(field_rate_rad_s=0., reference_index=1), calibration=cal)


def test_manual_field_center_allows_constant_capture_without_disc():
    source = ArraySource(np.full((2, 48, 48), 25.), timestamps=np.arange(2.))
    cfg = replace(config(field_rate_rad_s=0.), equatorial_radius_px=None, frame_preselection=False)
    result = stack_source(source, cfg)
    np.testing.assert_allclose(result.image, 25.)


@pytest.mark.parametrize("device", ["auto", "gpu"])
def test_geometry_reports_cpu_without_cuda_probe(monkeypatch, device):
    import planetrecon.backends.base as backends
    def forbidden():
        pytest.fail("CPU geometry must not probe unused CUDA operators")
    monkeypatch.setattr(backends, "probe_torch_cuda", forbidden)
    source = ArraySource(texture()[None], bit_depth=32)
    result = stack_source(source, replace(config(field_rate_rad_s=0.), device=device))
    assert result.backend == "cpu"
    assert result.precision == "float64"
    report = result.provenance["device_report"]
    assert report["requested"] == device and report["selected"] == "cpu"
    assert report["fallback"]


def test_cancel_before_pose_sampling_reads_no_frames():
    class Unreadable(ArraySource):
        def read_raw(self, index):
            pytest.fail("cancelled geometry should not read frames")
    source = Unreadable(texture()[None], bit_depth=32)
    result = stack_source(source, config(), should_cancel=lambda: True)
    assert result.incomplete and result.n_used == 0


def test_disc_moments_are_background_invariant_and_axes_are_semiaxes():
    yy, xx = np.indices((100, 120), dtype=float)
    disc = (((xx + .5 - 60) / 30)**2 + ((yy + .5 - 50) / 20)**2 <= 1).astype(float)
    fit = fit_disc_ellipse(disc)
    negative = fit_disc_ellipse(disc - 10.)
    for key in ("cx", "cy", "radius", "semi_major", "semi_minor"):
        assert fit[key] == pytest.approx(negative[key])
    assert fit["semi_major"] == pytest.approx(30., abs=.2)
    assert fit["semi_minor"] == pytest.approx(20., abs=.2)
    assert fit["radius"] == fit["semi_major"]


def test_bilinear_fill_at_partial_boundary_and_invalid_coordinates():
    src = np.full((2, 2), 10.)
    with np.errstate(all="raise"):
        out = bilinear_sample(src, np.array([0., np.nan, np.inf]), np.array([-.5, 0., 0.]), fill=2.)
        adj = bilinear_sample_adjoint(np.ones(2), np.array([np.nan, np.inf]), np.zeros(2), (2, 2))
    np.testing.assert_array_equal(out, [6., 2., 2.])
    assert not adj.any()


def test_unsupported_exposure_and_invalid_geometry_are_rejected():
    with pytest.raises(ValueError, match="quadrature"):
        config(exposure_s=1., freeze_mid_exposure=False)
    with pytest.raises(ValueError, match="cadence_s"):
        config(cadence_s=0.)
    with pytest.raises(ValueError, match="sub_obs_lat_rad"):
        config(sub_obs_lat_rad=np.pi)
    with pytest.raises(ValueError, match="sub_obs_lat_rad"):
        GlobeParams(20., sub_obs_lat_rad=np.pi)


def test_geometry_worker_retains_epoch_in_checkpoint_and_result(tmp_path):
    import time
    from planetrecon.jobs import start_stack_job, load_checkpoint
    path = tmp_path / "capture.ser"
    write_ser(path, np.stack([(texture() * 100).astype(np.uint8)] * 3))
    cfg = config(field_rate_rad_s=0., cadence_s=.1, reference_epoch_s=.15)
    job = start_stack_job(path, cfg, checkpoint_dir=tmp_path, job_id="geometry-review")
    events = []
    try:
        deadline = time.monotonic() + 20
        while job.state == "running" and time.monotonic() < deadline:
            events.extend(job.poll(timeout=.1))
        assert job.state == "completed", [(event.kind, event.payload.get("message")) for event in events]
        terminal = next(event.payload for event in events if event.kind == "completed")
        assert terminal["reference_epoch"] == "0.15"
        assert terminal["n_used"] == 3
        checkpoint = load_checkpoint(tmp_path / "geometry-review.npz", cfg)
        assert checkpoint["meta"]["reference_epoch"] == "0.15"
        assert checkpoint["meta"]["provenance"]["geometry"]["reference_epoch_s"] == .15
    finally:
        job.close()


def test_degeneracy_warning_does_not_claim_supplied_rate_is_missing():
    yy, xx = np.indices((48, 48), dtype=float)
    disc = np.clip(1 - ((xx + .5 - 24)**2 + (yy + .5 - 24)**2) / 16**2, 0., None)
    source = ArraySource(np.stack([disc, disc]), bit_depth=32, timestamps=np.arange(2.))
    cfg = replace(config(field_rate_rad_s=0.), geometry_mode="surface", surface_rate_rad_s=.1)
    result = stack_source(source, cfg)
    assert any("using supplied surface rate" in warning for warning in result.warnings)
    assert not any("not supplied" in warning for warning in result.warnings)
    field_result = stack_source(source, config(field_rate_rad_s=0.))
    assert not any("spin_unconstrained" in warning for warning in field_result.warnings)
