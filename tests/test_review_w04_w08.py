"""Regressions for capture validation, numerical coverage and owned workers."""

import json
import queue
import struct
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.ndimage import shift as ndshift

from planetrecon.backends.base import Backend, probe_torch_cuda, select_backend
from planetrecon.calibration import Calibration
from planetrecon.detector import cfa_accumulate, cfa_labels
from planetrecon.io.ser import SER_HEADER_SIZE, SERSource, write_ser
from planetrecon.io.source import ArraySource, open_source
from planetrecon.jobs import JobEvent, _put_event, load_checkpoint, start_stack_job
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig


def cpu_config(**kwargs):
    return ReconstructionConfig(device="cpu", threads=2, **kwargs)


@pytest.mark.parametrize("rgb", [False, True])
def test_shifted_stack_keeps_edge_brightness(monkeypatch, rgb):
    monkeypatch.setattr(Backend, "phase_correlation", lambda *_: (2.5, -1.5))
    shape = (2, 12, 12, 3) if rgb else (2, 12, 12)
    result = stack_source(ArraySource(np.full(shape, 20.0), color_mode="RGB" if rgb else "mono"), cpu_config())
    np.testing.assert_allclose(result.image, 20.0, atol=1e-12)
    assert np.min(result.coverage) < np.max(result.coverage)


@pytest.mark.parametrize("pattern", ["RGGB", "GRBG", "GBRG", "BGGR"])
def test_cfa_coverage_is_per_channel_and_adjoint(pattern):
    rng = np.random.default_rng(17)
    latent = rng.normal(size=(9, 11, 3))
    data = rng.normal(size=(9, 11))
    labels = cfa_labels(9, 11, pattern)
    displacement = (0.4, -0.7)
    forward = np.zeros_like(data)
    for c, name in enumerate("RGB"):
        forward += (labels == name) * ndshift(latent[..., c],
                    shift=(displacement[1], displacement[0]), order=1,
                    prefilter=False, mode="grid-constant")
    adjoint, _ = cfa_accumulate(data, displacement, pattern)
    assert np.vdot(forward, data) == pytest.approx(np.vdot(latent, adjoint), abs=1e-12)
    result = stack_source(ArraySource(np.ones((1, 9, 11)), color_mode=pattern), cpu_config())
    assert result.validity.shape == result.image.shape
    assert np.all(result.validity.sum(axis=2) == 1)
    np.testing.assert_array_equal(result.validity, result.coverage > 0)


def test_rejection_progress_units_and_snapshot_independence():
    frames = np.full((4, 12, 12), 10.0)
    frames[0] = np.nan
    frames[2] = 255.0
    events = []
    result = stack_source(ArraySource(frames, bit_depth=8), cpu_config(batch_frames=1),
                          calibration=Calibration(gain_e_per_adu=2),
                          on_event=lambda result, info: events.append((result, info)))
    assert (result.n_used, result.n_rejected, result.units) == (2, 2, "e-")
    np.testing.assert_allclose(result.image, 20)
    assert events[-1][1]["n_processed"] == 4
    assert events[-1][1]["n_total"] == 4
    assert events[1][0].coverage.max() < result.coverage.max()
    assert all(event.units == "e-" for event, _ in events)
    assert result.provenance["reference_index"] == 1
    with pytest.raises(ValueError, match="no usable"):
        stack_source(ArraySource(frames[:1]), cpu_config())


def test_reference_selection_and_calibration_saturation(monkeypatch):
    frames = np.stack([np.full((10, 10), value) for value in (10., 20., 30.)])
    refs = []
    def correlate(self, reference, frame):
        refs.append(reference.copy())
        return (0., 0.)
    monkeypatch.setattr(Backend, "phase_correlation", correlate)
    result = stack_source(ArraySource(frames), cpu_config(reference_index=1),
                          calibration=Calibration(saturate_adu=25))
    assert result.n_rejected == 1
    assert result.provenance["reference_index"] == 1
    assert all(np.all(ref == 20) for ref in refs)


def test_hdf5_electrons_are_not_adc_clipped(tmp_path):
    path = tmp_path / "observed.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("frames/feature_observed_e", data=np.full((1, 10, 10), 100000.))
    with open_source(path) as source:
        result = stack_source(source, cpu_config())
    assert result.units == "e-"
    np.testing.assert_allclose(result.image, 100000.)


@pytest.mark.parametrize("offset,value", [(26, 0), (30, -2), (34, 17), (38, -1), (22, 5)])
def test_ser_rejects_invalid_numeric_headers(tmp_path, offset, value):
    path = write_ser(tmp_path / "bad.ser", np.zeros((1, 8, 8), np.uint8))
    blob = bytearray(path.read_bytes())
    struct.pack_into("<i", blob, offset, value)
    path.write_bytes(blob)
    with pytest.raises(ValueError):
        SERSource(path)


def test_ser_trailers_empty_capture_and_overrides(tmp_path):
    path = write_ser(tmp_path / "trailer.ser", np.zeros((2, 8, 8), np.uint8), timestamps=np.array([1, 2]))
    blob = path.read_bytes()
    path.write_bytes(blob[:-3])
    with pytest.raises(ValueError, match="trailer"):
        SERSource(path)
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="Bayer"):
        SERSource(path, bayer_override="typo")
    with pytest.raises(ValueError, match="endian"):
        SERSource(path, endian_override="typo")
    with SERSource(path) as source:
        np.testing.assert_array_equal(source.timestamps(), [1, 2])
    empty = write_ser(tmp_path / "empty.ser", np.zeros((0, 8, 8), np.uint8))
    assert empty.stat().st_size == SER_HEADER_SIZE
    with SERSource(empty) as source:
        assert source.n_frames() == 0
        with pytest.raises(ValueError, match="requires frames"):
            stack_source(source, cpu_config())
    path = write_ser(tmp_path / "cmy.ser", np.ones((1, 8, 8), np.uint8), color_id=16)
    with SERSource(path) as source:
        with pytest.raises(ValueError, match="unsupported.*color"):
            stack_source(source, cpu_config())


@pytest.mark.parametrize("kwargs", [{"batch_frames": 0}, {"reference_index": -1},
    {"device": "typo"}, {"max_shift_px": float("nan")}, {"endian_convention": "typo"},
    {"max_ram_bytes": 1000}, {"max_vram_bytes": 1000}])
def test_config_rejects_ignored_or_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ReconstructionConfig(**kwargs)


def test_gpu_probe_success_and_query_failure_are_safe(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "test GPU",
                           get_device_capability=lambda _: (12, 0),
                           get_arch_list=lambda: ["sm_120"], synchronize=lambda: None)
    scalar = SimpleNamespace(item=lambda: 1.)
    tensor = SimpleNamespace(abs=lambda: SimpleNamespace(sum=lambda: scalar))
    torch = SimpleNamespace(cuda=cuda, float32="float32", float64="float64", randn=lambda *a, **kw: tensor,
                            fft=SimpleNamespace(fft2=lambda x: tensor))
    monkeypatch.setitem(sys.modules, "torch", torch)
    report = probe_torch_cuda()
    assert report.selected == "gpu" and report.warnings == []
    json.dumps(report.__dict__)
    monkeypatch.setitem(sys.modules, "planetrecon.backends.torch_accel",
                        SimpleNamespace(TorchBackend=lambda: Backend()))
    _, report = select_backend("auto", threads=2)
    assert report.requested == "auto"
    def fail(_):
        raise RuntimeError("driver unavailable")
    cuda.get_device_name = fail
    report = probe_torch_cuda()
    assert report.selected == "cpu" and "capability_query_failed" in report.reason


def test_unsupported_arch_does_not_launch_cuda(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "RTX 5070",
                           get_device_capability=lambda _: (12, 0),
                           get_arch_list=lambda: ["sm_90"])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__="2.6"))
    report = probe_torch_cuda()
    assert report.selected == "cpu" and report.fallback
    assert "sm_120" in report.reason


def test_ui_updates_never_block_when_queue_is_full():
    q = queue.Queue(maxsize=1)
    q.put(JobEvent("job", 1, "error"))
    _put_event(q, JobEvent("job", 2, "progress"))
    _put_event(q, JobEvent("job", 3, "preview"))
    assert q.get_nowait().kind == "error"


def test_full_event_queue_does_not_stall_stack_or_cancellation(tmp_path):
    rng = np.random.default_rng(19)
    frame = rng.integers(10, 100, size=(12, 12), dtype=np.uint8)
    path = write_ser(tmp_path / "job.ser", np.repeat(frame[None], 20, axis=0))
    cfg = cpu_config(batch_frames=1)
    handle = start_stack_job(path, cfg, job_id="review", queue_size=2, checkpoint_dir=tmp_path / "ckpt")
    checkpoint = tmp_path / "ckpt/review.npz"
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if checkpoint.exists() and load_checkpoint(checkpoint, cfg)["meta"]["n_used"] == 20:
                break
            time.sleep(0.02)
        else:
            pytest.fail("worker stalled on unconsumed progress events")
        handle.cancel(grace_s=3)
        assert not handle.process.is_alive()
        assert handle.process.exitcode == 0  # cooperative, no terminate/kill
        saved = load_checkpoint(checkpoint, cfg)
        np.testing.assert_allclose(saved["image"], frame)
        assert saved["meta"]["provenance"]["source"]["path"] == str(path)
        assert not checkpoint.with_suffix(".json").exists()
    finally:
        handle.close()


def test_thread_environment_and_cli_keep_requested_limit(monkeypatch):
    from planetrecon.cli import main
    from planetrecon.runtime import merge_thread_env, thread_env
    assert set(thread_env(100).values()) == {"32"}
    assert merge_thread_env({}, 100)["PLANETRECON_THREADS"] == "32"
    monkeypatch.setenv("PLANETRECON_THREADS", "2")
    assert main(["probe-device", "--device", "cpu"]) == 0
    assert ReconstructionConfig().threads == 2


@pytest.mark.parametrize("saturated", [False, True])
def test_worker_delivers_terminal_result_or_error(tmp_path, saturated):
    frame = np.full((12, 16), 255 if saturated else 25, dtype=np.uint8)
    path = write_ser(tmp_path / "terminal.ser", np.repeat(frame[None], 3, axis=0))
    handle = start_stack_job(path, cpu_config(batch_frames=1), queue_size=2)
    events = []
    try:
        deadline = time.monotonic() + 15
        while handle.state in ("queued", "running") and time.monotonic() < deadline:
            events.extend(handle.poll(timeout=0.05))
        assert handle.state == ("failed" if saturated else "completed")
        terminal = events[-1]
        if saturated:
            assert "no usable frames" in terminal.payload["message"]
        else:
            assert terminal.payload["n_used"] == 3
            assert not terminal.payload["incomplete"]
            np.testing.assert_allclose(terminal.payload["image"], frame)
            assert terminal.payload["units"] == "adu"
        handle.close()
        assert not handle.process.is_alive()
    finally:
        handle.close()


def test_unexpected_clean_worker_exit_is_reported():
    from planetrecon.jobs import JobHandle
    handle = JobHandle("test", SimpleNamespace(is_alive=lambda: False, exitcode=0),
                       queue.Queue(), SimpleNamespace(is_set=lambda: False), state="running")
    events = handle.poll()
    assert handle.state == "failed"
    assert events[-1].kind == "error"


@pytest.mark.skipif(__import__("importlib").util.find_spec("PySide6") is None,
                    reason="Qt6 is not installed")
def test_qt_close_owns_worker_and_preserves_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from planetrecon.gui.app import MainWindow, create_app

    path = write_ser(tmp_path / "gui.ser", np.full((20, 12, 12), 10, dtype=np.uint8))
    app = create_app(["review"])
    cfg = cpu_config(batch_frames=1, bayer_override="BGGR", reference_index=1, reject_saturated=False)
    win = MainWindow(path, config=cfg)
    captured = []
    def start(path, config):
        captured.append(config)
        return start_stack_job(path, config)
    monkeypatch.setattr("planetrecon.gui.app.start_stack_job", start)
    win.show()
    try:
        win._run()
        handle = win.job
        assert captured[0] == cfg
        win.window.close()
        app.processEvents()
        assert win.job is None
        assert not handle.process.is_alive()
    finally:
        win._shutdown()
        win.window.close()
