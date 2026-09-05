import json
from pathlib import Path

import numpy as np
import pytest

from planetrecon.backends.base import probe_torch_cuda, select_backend
from planetrecon.calibration import Calibration, apply_calibration
from planetrecon.detector import cfa_accumulate, extract_green_proxy, is_bayer
from planetrecon.io.hdf5_source import HDF5ObservedSource
from planetrecon.io.ser import COLOR_BGR, COLOR_RGB, COLOR_RGGB, write_ser
from planetrecon.io.source import ArraySource, open_source
from planetrecon.jobs import load_checkpoint, start_stack_job
from planetrecon.pipeline.align import phase_correlation_shift
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult


def test_config_schema_rejection():
    with pytest.raises(ValueError, match="schema"):
        ReconstructionConfig.from_dict(
            {"schema_name": "not-this", "schema_version": "1.0"}
        )
    with pytest.raises(ValueError, match="schema_version"):
        ReconstructionConfig.from_dict(
            {"schema_name": "planetrecon-reconstruction-config", "schema_version": "9.9"}
        )


def test_config_cache_key_stable():
    cfg = ReconstructionConfig(device="cpu", threads=4)
    a = cfg.cache_key("ser://x")
    b = cfg.cache_key("ser://x")
    c = cfg.cache_key("ser://y")
    assert a == b and a != c


def test_array_source_has_no_truth():
    frames = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
    src = ArraySource(frames, color_mode="mono")
    assert not src.has_truth()
    assert src.n_frames() == 2
    assert src.read_raw(0).shape == (8, 8)
    with pytest.raises(IndexError):
        src.read_raw(9)


def test_ser_roundtrip_mono_and_bayer(tmp_path):
    mono = np.arange(3 * 6 * 8, dtype=np.uint8).reshape(3, 6, 8)
    path = write_ser(tmp_path / "mono.ser", mono, color_id=0, pixel_depth=8)
    with open_source(path) as src:
        meta = src.metadata()
        assert meta.color_mode == "mono"
        assert meta.n_frames == 3
        assert np.array_equal(src.read_raw(1), mono[1])
        assert src.timestamps() is None

    raw = np.zeros((2, 8, 8), dtype=np.uint16)
    raw[0, 0::2, 0::2] = 100
    raw[0, 0::2, 1::2] = 200
    raw[0, 1::2, 0::2] = 210
    raw[0, 1::2, 1::2] = 50
    raw[1] = raw[0] + 1
    path = write_ser(tmp_path / "rggb.ser", raw, color_id=COLOR_RGGB, pixel_depth=16)
    with open_source(path) as src:
        assert src.color_mode() == "RGGB"
        assert src.read_raw(0).dtype == np.uint16
        assert src.read_raw(0)[0, 0] == 100


def test_ser_endian_conventions(tmp_path):
    frames = np.array([[[0x0102, 0x1100], [0x00FF, 0x8000]]], dtype=np.uint16)
    le = write_ser(tmp_path / "le.ser", frames, pixel_depth=16, little_endian_flag=0)
    be = write_ser(
        tmp_path / "be.ser",
        frames,
        pixel_depth=16,
        little_endian_flag=1,
        endian_convention="ecosystem",
    )
    with open_source(le) as src:
        assert src.metadata().endian == "little"
        assert src.read_raw(0)[0, 0] == 0x0102
    with open_source(be) as src:
        assert src.metadata().endian == "big"
        assert src.read_raw(0)[0, 0] == 0x0102
    with open_source(le, endian_override="big") as src:
        # Forced opposite byte swap of a little-endian file.
        assert src.read_raw(0)[0, 0] != 0x0102


def test_ser_rgb_bgr_and_unknown_id(tmp_path):
    rgb = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    path = write_ser(tmp_path / "rgb.ser", rgb, color_id=COLOR_RGB)
    with open_source(path) as src:
        assert src.color_mode() == "RGB"
        assert src.read_raw(0).shape == (4, 4, 3)
        assert src.read_raw(0)[0, 0, 0] == 10
    bgr = rgb[..., ::-1]
    path = write_ser(tmp_path / "bgr.ser", bgr, color_id=COLOR_BGR)
    with open_source(path) as src:
        got = src.read_raw(0)
        assert got[0, 0, 0] == 10  # swapped back to RGB order
    with pytest.raises(ValueError, match="recognised"):
        write_ser(tmp_path / "bad.ser", np.zeros((1, 2, 2), np.uint8), color_id=7)


def test_ser_truncated_and_recover(tmp_path):
    frames = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    path = write_ser(tmp_path / "full.ser", frames)
    data = path.read_bytes()
    cut = tmp_path / "cut.ser"
    # header + 1.5 frames
    from planetrecon.io.ser import SER_HEADER_SIZE

    frame_bytes = 16
    cut.write_bytes(data[: SER_HEADER_SIZE + frame_bytes + 8])
    with pytest.raises(ValueError, match="truncated"):
        open_source(cut)
    with open_source(cut, recover_complete_frames=True) as src:
        assert src.n_frames() == 1


def test_hdf5_source_rejects_truth(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="truth"):
        HDF5ObservedSource(tmp_path / "x.h5", truth=True)


def test_phase_correlation_recovers_shift():
    yy, xx = np.indices((32, 32))
    obj = np.exp(-0.5 * ((xx - 16) ** 2 + (yy - 16) ** 2) / 3.0**2)
    from scipy.ndimage import shift as ndshift

    moved = ndshift(obj, shift=(2.0, -3.0), order=1, prefilter=False)
    sx, sy = phase_correlation_shift(obj, moved)
    assert sx == pytest.approx(-3.0, abs=0.2)
    assert sy == pytest.approx(2.0, abs=0.2)


def test_cfa_joint_keeps_both_greens():
    raw = np.zeros((8, 8), dtype=np.float64)
    raw[0::2, 1::2] = 10.0  # G in RGGB
    raw[1::2, 0::2] = 12.0
    raw[0::2, 0::2] = 5.0
    raw[1::2, 1::2] = 3.0
    rgb, weight = cfa_accumulate(raw, (0.0, 0.0), "RGGB")
    assert is_bayer("RGGB")
    assert rgb[..., 1][0, 1] == pytest.approx(10.0)
    assert rgb[..., 1][1, 0] == pytest.approx(12.0)
    proxy = extract_green_proxy(raw, "RGGB")
    assert proxy[0, 1] == pytest.approx(10.0)


def test_calibration_missing_is_labelled():
    frame = np.ones((4, 4)) * 10
    out, info = apply_calibration(frame, None)
    assert info["mode"] == "approximate-noise"
    cal = Calibration(gain_e_per_adu=2.0, saturate_adu=8)
    out, info = apply_calibration(frame, cal)
    assert info["mode"] == "calibrated"
    assert info["saturated"]
    assert out.mean() == pytest.approx(20.0)


def test_stack_array_mono_and_cfa():
    rng = np.random.default_rng(0)
    base = np.clip(40 + 80 * np.exp(-0.5 * ((np.indices((24, 24))[1] - 12) ** 2 + (np.indices((24, 24))[0] - 12) ** 2) / 8.0), 0, 255)
    frames = np.stack([base, np.roll(base, 1, axis=1), np.roll(base, -1, axis=0)])
    src = ArraySource(frames.astype(np.float64), color_mode="mono", bit_depth=16)
    cfg = ReconstructionConfig(device="cpu", threads=2, batch_frames=2)
    result = stack_source(src, cfg)
    assert result.n_used == 3
    assert result.channel_order == "mono"
    assert result.image.shape == (24, 24)
    assert result.backend == "cpu"

    cfa = np.zeros((3, 16, 16), dtype=np.float64)
    for k in range(3):
        cfa[k, 0::2, 0::2] = 40 + k
        cfa[k, 0::2, 1::2] = 80
        cfa[k, 1::2, 0::2] = 82
        cfa[k, 1::2, 1::2] = 20
    src = ArraySource(cfa, color_mode="RGGB", bit_depth=16)
    result = stack_source(src, cfg)
    assert result.channel_order == "RGB"
    assert result.image.shape == (16, 16, 3)
    assert result.provenance["demosaic_first"]["label"] == "comparison"


def test_auto_device_falls_back_when_arch_unsupported(monkeypatch):
    from planetrecon.backends.base import DeviceReport

    def unsupported():
        return DeviceReport("gpu", "cpu", ["cpu"], True, "cuda_arch_unsupported:sm_120", "RTX 5070")

    monkeypatch.setattr("planetrecon.backends.base.probe_torch_cuda", unsupported)
    backend, report = select_backend("auto", threads=2)
    assert backend.name == "cpu"
    gpu = unsupported()
    assert gpu.fallback
    assert "sm_120" in gpu.reason or "cuda" in gpu.reason or "torch" in gpu.reason
    backend, report = select_backend("gpu", threads=2)
    assert report.fallback
    assert report.selected == "cpu"
    assert report.warnings


def test_job_cancel_and_checkpoint(tmp_path):
    frames = (np.arange(12 * 16 * 16, dtype=np.uint8).reshape(12, 16, 16) % 200)
    path = write_ser(tmp_path / "job.ser", frames)
    cfg = ReconstructionConfig(device="cpu", threads=2, batch_frames=2)
    handle = start_stack_job(path, cfg, checkpoint_dir=tmp_path / "ckpt")
    handle.cancel(grace_s=3.0)
    assert handle.state == "cancelled"
    assert not handle.process.is_alive()
    handle.close()
    result = ReconstructionResult(
        image=np.ones((4, 4)),
        coverage=np.ones((4, 4)),
        validity=np.ones((4, 4), dtype=bool),
        units="adu",
        channel_order="mono",
        backend="cpu",
        precision="float64",
        stage="baseline",
        incomplete=True,
        n_used=1,
    )
    from planetrecon.jobs import _write_checkpoint

    _write_checkpoint(str(tmp_path / "ckpt2"), "t", cfg, result)
    loaded = load_checkpoint(tmp_path / "ckpt2" / "t.npz", cfg)
    assert loaded["meta"]["n_used"] == 1
    other = ReconstructionConfig(device="cpu", threads=4)
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(tmp_path / "ckpt2" / "t.npz", other)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("PySide6") is None,
    reason="PySide6 is required for the Qt shell test",
)
def test_qt_shell_offscreen_preview_and_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from planetrecon.gui.app import MainWindow, create_app

    frames = np.arange(6 * 12 * 12, dtype=np.uint8).reshape(6, 12, 12)
    path = write_ser(tmp_path / "gui.ser", frames)
    app = create_app(["planetrecon-test"])
    win = MainWindow(path=path, config=ReconstructionConfig(device="cpu", threads=2, batch_frames=2))
    win.show()
    win._run()
    app.processEvents()
    win._cancel()
    app.processEvents()
    assert win.job is None or win.job.state in ("cancelled", "running", "completed", "failed")
    if win.job is not None:
        win.job.close()
    win.window.close()
