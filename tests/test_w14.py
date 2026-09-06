"""W14 Qt workflow, live result ownership, calibration and export regressions."""
from dataclasses import replace
import json
import os
import queue
import threading
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
import tifffile

from planetrecon.calibration import Calibration, load_calibration
from planetrecon.export import ExportCancelled, ExportConfig, export_result
from planetrecon.io.ser import write_ser
from planetrecon.io.source import ArraySource
from planetrecon.jobs import JobEvent, JobHandle, result_payload, start_stack_job
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult


def config(**kw):
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=1, **kw)


def result(value=2, incomplete=True):
    return ReconstructionResult(np.full((514, 12, 3), value, dtype=float),
        np.ones((514, 12, 3)), np.ones((514, 12, 3), bool), 'adu', 'RGB', 'cpu', 'float64',
        'baseline' if incomplete else 'final', incomplete, reference_epoch='2.5', n_used=1,
        provenance={'source': {'path': 'original.ser'}}, warnings=['example warning'],
        layer_coverage={'globe': np.ones((514,12)), 'ring': np.zeros((514,12))})


@pytest.fixture
def gui():
    pytest.importorskip('PySide6')
    from planetrecon.gui.app import MainWindow, create_app
    app = create_app(['w14-test'])
    win = MainWindow(config=config())
    win.show()
    yield app, win
    win._shutdown()
    if win.export_worker is not None:
        win.export_worker.wait(5000)
        app.processEvents()
    win.window.close()
    app.processEvents()


def pump(app, condition, timeout=10):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert condition(), 'Qt condition timed out'


def test_cuda_budget_control_preserves_mib_units(gui):
    _, win = gui
    fields = win.controls.fields
    fields['device'].setCurrentText('gpu')
    fields['max_vram_bytes'].setText('64.5')
    cfg = win.controls.configuration()
    assert cfg.max_vram_bytes == int(64.5 * 1024**2)
    from planetrecon.gui.controls import ConfigControls
    restored = ConfigControls(cfg)
    assert restored.configuration() == cfg
    restored.deleteLater()
    fields['max_vram_bytes'].setText('0')
    with pytest.raises(ValueError, match='positive'):
        win.controls.configuration()


def test_config_controls_preserve_units_and_optional_state(gui):
    _, win = gui
    fields = win.controls.fields
    fields['geometry_mode'].setCurrentText('saturn')
    for key, text in dict(sub_obs_lat_rad='-20', field_rate_rad_s='1.2', equatorial_radius_px='12',
                          ring_inner_radius_px='16', ring_outer_radius_px='26', gain_e_per_adu='2.5',
                          moon_x='12', moon_y='20', moon_radius_px='3', moon_vy_px_s='-.3').items():
        fields[key].setText(text)
    fields['bayer_override'].setCurrentText('BGGR')
    cfg = win.controls.configuration()
    assert cfg.sub_obs_lat_rad == pytest.approx(np.radians(-20))
    assert cfg.field_rate_rad_s == pytest.approx(np.radians(1.2))
    assert cfg.bayer_override == 'BGGR' and cfg.gain_e_per_adu == 2.5
    assert cfg.moon_vy_px_s == -.3
    assert cfg.equatorial_radius_px == 12 and cfg.ring_inner_radius_px == 16
    assert cfg.ring_outer_radius_px == 26 and cfg.moon_radius_px == 3
    fields['ring_inner_radius_px'].setText('bad text')
    fields['geometry_mode'].setCurrentText('none')
    cfg = win.controls.configuration()
    assert cfg.ring_inner_radius_px is None and cfg.moon_y is None


def test_calibration_tables_are_loaded_on_raw_lattice(tmp_path):
    frame = np.arange(96.).reshape(8,12) + 30
    bias = np.full_like(frame, 10)
    path = tmp_path/'bias.npy'
    np.save(path, bias)
    cfg = config(bias_path=str(path), gain_e_per_adu=2.)
    r = stack_source(ArraySource(frame[None], bit_depth=32), cfg)
    np.testing.assert_allclose(r.image, (frame-bias)*2)
    assert r.units == 'e-' and r.provenance['calibration']['bias']['sha256']
    with pytest.raises(ValueError, match='either'):
        stack_source(ArraySource(frame[None]), cfg, calibration=Calibration())
    np.save(path, np.zeros_like(frame))
    with pytest.raises(ValueError, match='positive'):
        load_calibration(config(flat_path=str(path)), frame.shape)
    np.save(path, np.ones((8,1)))
    with pytest.raises(ValueError, match='shape'):
        load_calibration(cfg, frame.shape)


@pytest.mark.parametrize('kw', [dict(gain_e_per_adu=0), dict(read_noise_e=-1), dict(saturate_adu=np.inf),
                                  dict(flat_path=''), dict(gain_e_per_adu=True)])
def test_calibration_config_rejects_invalid_values(kw):
    with pytest.raises(ValueError):
        config(**kw)


def test_owned_worker_only_sends_one_unacknowledged_full_snapshot(tmp_path):
    frame = np.arange(96, dtype=np.uint16).reshape(8,12) + 100
    path = write_ser(tmp_path/'bounded.ser', np.stack([frame]*20))
    job = start_stack_job(path, config(), queue_size=32)
    job.snapshot_request.set()
    events = []
    try:
        deadline = time.monotonic()+10
        while job.state == 'running' and time.monotonic() < deadline:
            events.extend(job.poll(.05))
        assert job.state == 'completed'
        snapshots = [e for e in events if e.kind == 'snapshot']
        assert len(snapshots) == 1
        snap = snapshots[0].payload
        assert snap['image'].shape == frame.shape and snap['incomplete'] and snap['n_used'] == 1
        completed = next(e.payload for e in events if e.kind == 'completed')
        assert not completed['incomplete'] and completed['n_used'] == 20
    finally:
        job.close()


def test_gui_real_source_inspection_calibration_and_completion(gui, tmp_path):
    app, win = gui
    frame = np.arange(96, dtype=np.uint16).reshape(8,12) + 100
    win.path = write_ser(tmp_path/'real.ser', np.stack([frame]*4))
    win._inspect()
    pump(app, lambda: win.job is None)
    assert win.input_image.shape == frame.shape and '4 frames' in win.source_label.text()
    assert win.last_result is None
    win.controls.fields['gain_e_per_adu'].setText('2')
    win._run()
    pump(app, lambda: win.job is None)
    assert not win.last_result.incomplete and win.last_result.n_used == 4
    np.testing.assert_allclose(win.last_result.image, frame*2)
    assert win.last_result.units == 'e-' and win.save_btn.isEnabled()
    assert win.progress.value() == 100
    assert not win.last_result.image.flags.writeable


def test_gui_resumes_partial_checkpoint_and_preserves_result_on_mismatch(gui,tmp_path):
    app,win=gui
    frame=np.arange(96,dtype='u2').reshape(8,12)+100
    win.path=write_ser(tmp_path/'in.ser',np.stack([frame]*5))
    state=tmp_path/'state.npz';cfg=win.controls.configuration()
    from planetrecon.io.ser import SERSource
    with SERSource(win.path) as source:
        expected=stack_source(source,cfg)
    cancel=False
    def event(result,info):
        nonlocal cancel
        cancel=result.n_used>=2
    with SERSource(win.path) as source:
        stack_source(source,cfg,state_checkpoint=state,on_event=event,should_cancel=lambda:cancel)
    win.checkpoint_path.setText(str(state));win.resume_check.setChecked(True)
    win._run()
    assert not win.checkpoint_path.isEnabled()
    pump(app,lambda:win.job is None)
    assert not win.error.text() and win.last_result.n_used==5
    assert win.last_result.provenance['resumed_from_frame']==2
    np.testing.assert_array_equal(win.last_result.image,expected.image)
    prior=win.last_result
    win.controls.fields['max_shift_px'].setText('12')
    win._run();pump(app,lambda:win.job is None)
    assert 'mismatch' in win.error.text() and win.last_result is prior


def test_worker_failures_preserve_last_result_and_allow_rerun(gui, tmp_path):
    app, win = gui
    old = result(incomplete=False)
    win._accept_result(result_payload(old))
    win.path = tmp_path/'missing.ser'
    win._run()
    pump(app, lambda: win.job is None)
    assert win.last_result.image[0,0,0] == 2 and win.save_btn.isEnabled()
    assert 'Processing failed' in win.error.text() and 'original.ser' in win.result_label.text()
    win.path = write_ser(tmp_path/'good.ser', np.full((2,8,12), 20, np.uint16))
    win._run()
    pump(app, lambda: win.job is None)
    assert not win.error.text() and win.last_result.image[0,0] == 20


def test_save_retains_full_intermediate_snapshot_during_updates(gui, tmp_path, monkeypatch):
    app, win = gui
    from PySide6.QtCore import QTimer
    import planetrecon.gui.app as module
    entered, release = threading.Event(), threading.Event()
    original = module.export_result
    def slow_export(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'export_result', slow_export)
    r = result(value=3)
    win._accept_result(result_payload(r))
    assert win.preview.image.shape[0] < r.image.shape[0]
    dest = tmp_path/'live.tif'
    assert win.save_result(dest)
    assert entered.wait(2)
    ticks = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    try:
        win._accept_result(result_payload(result(value=8, incomplete=False)))
        pump(app, lambda: len(ticks) >= 5)
        assert win.export_worker is not None
        release.set()
        pump(app, lambda: win.export_worker is None)
    finally:
        timer.stop()
        release.set()
    with tifffile.TiffFile(dest) as tf:
        image = tf.pages[0].asarray()
        meta = json.loads(tf.pages[0].description)
    assert image.shape == (514,12,3) and np.all(image == 3)
    assert meta['result']['incomplete'] and meta['result']['reference_epoch'] == '2.5'
    assert win.last_result.image[0,0,0] == 8


def test_save_failure_overwrite_decline_and_retry(gui, tmp_path, monkeypatch):
    app, win = gui
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    win._accept_result(result_payload(result()))
    path = tmp_path/'exists.tif'
    path.write_bytes(b'old image')
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(path), 'TIFF'))
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **kw: QMessageBox.StandardButton.No)
    win._choose_save()
    assert win.export_worker is None and path.read_bytes() == b'old image'
    assert win.save_result(path)
    pump(app, lambda: win.export_worker is None)
    assert 'Save failed' in win.save_status.text() and win.save_btn.isEnabled()
    assert path.read_bytes() == b'old image'
    assert win.save_result(path, overwrite=True)
    pump(app, lambda: win.export_worker is None)
    assert 'Saved' in win.save_status.text()


def test_cancelled_export_does_not_publish_and_closes_thread(gui, tmp_path, monkeypatch):
    app, win = gui
    import planetrecon.gui.app as module
    release = threading.Event()
    original = module.export_result
    def slow(*args, **kwargs):
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'export_result', slow)
    win._accept_result(result_payload(result()))
    dest = tmp_path/'cancel.tif'
    win.save_result(dest)
    worker = win.export_worker
    win.window.close()
    assert win.closing and win.window.isVisible()
    release.set()
    pump(app, lambda: win.export_worker is None and not win.window.isVisible())
    assert 'cancelled' in win.save_status.text().lower()
    assert not dest.exists()


def test_export_cancellation_after_encoding_preserves_existing_image(tmp_path, monkeypatch):
    import planetrecon.export as module
    p = tmp_path/'cancel.tif'
    p.write_bytes(b'old')
    cancel = threading.Event()
    original = module._write_tiff
    def write(*a, **kw):
        original(*a, **kw)
        cancel.set()
    monkeypatch.setattr(module, '_write_tiff', write)
    with pytest.raises(ExportCancelled):
        export_result(result(), p, overwrite=True, should_cancel=cancel.is_set)
    assert p.read_bytes() == b'old' and list(tmp_path.iterdir()) == [p]


def test_display_levels_stay_fixed_and_views_do_not_mutate_science(gui):
    _, win = gui
    win._accept_result(result_payload(result(value=2)))
    levels = win.black.value(), win.white.value()
    r = result(value=100)
    r.validity[0,0,0] = False
    win._accept_result(result_payload(r))
    assert (win.black.value(), win.white.value()) == levels
    for view in ('Coverage', 'Validity', 'Globe coverage', 'Ring coverage', 'Result'):
        win.view.setCurrentText(view)
        win.channel.setCurrentText('G')
        win.zoom.setCurrentText('200%')
    np.testing.assert_array_equal(win.last_result.image, r.image)
    assert 'clipped' in win.histogram.text() and 'example warning' in win.warnings.text()


def test_job_rejects_wrong_identity_and_duplicate_events():
    q = queue.Queue()
    q.put(JobEvent('old', 100, 'error'))
    q.put(JobEvent('current', 1, 'progress'))
    q.put(JobEvent('current', 1, 'error'))
    process = SimpleNamespace(is_alive=lambda: True)
    job = JobHandle('current', process, q, threading.Event(), state='running')
    events = job.poll()
    assert [e.kind for e in events] == ['progress'] and job.state == 'running'


def test_cancel_does_not_read_abandoned_snapshot_pipe():
    class BrokenQueue:
        def get(self, **kw):
            raise AssertionError('cancel must not read a possibly partial pickle')
    cancel = threading.Event()
    cancel.set()
    job = JobHandle('job', SimpleNamespace(is_alive=lambda: False), BrokenQueue(), cancel, state='running')
    assert [e.kind for e in job.poll()] == ['cancelled']


def test_gui_cancellation_is_nonblocking_and_releases_worker(gui, tmp_path):
    app, win = gui
    frame = np.arange(128*128, dtype=np.uint16).reshape(128,128)
    win.path = write_ser(tmp_path/'long.ser', np.stack([frame]*300))
    win._run()
    job = win.job
    started = time.monotonic()
    win._cancel()
    assert time.monotonic()-started < .1
    pump(app, lambda: win.job is None, timeout=5)
    assert not job.process.is_alive()


def test_gui_saturn_geometry_and_layer_export(gui, tmp_path):
    app, win = gui
    from planetrecon.geometry.globe import GlobeParams
    from planetrecon.geometry.rings import RingParams
    from planetrecon.geometry.pose import FramePose
    from planetrecon.geometry.saturn import render_saturn
    frame = render_saturn(48,64,FramePose(0,0,32,24),
        GlobeParams(12,flattening=.1,sub_obs_lat_rad=.4), RingParams(16,26),
        lambda lon,lat:50.,ring_tex=lambda r,a:100.)
    win.path = write_ser(tmp_path/'saturn.ser', np.stack([frame.astype(np.uint16)]*2))
    f = win.controls.fields
    f['geometry_mode'].setCurrentText('saturn')
    for key, val in dict(equatorial_radius_px=12, flattening=.1, sub_obs_lat_rad=np.degrees(.4),
        ring_inner_radius_px=16, ring_outer_radius_px=26, field_rate_rad_s=0,
        surface_rate_rad_s=0, field_center_x=32, field_center_y=24, cadence_s=.1).items():
        f[key].setText(str(val))
    win._run()
    pump(app, lambda: win.job is None)
    assert set(win.last_result.layer_coverage) == {'globe', 'ring'}
    win.view.setCurrentText('Ring coverage')
    assert win._view_data()[0].max() > 0
    dest = tmp_path/'saturn.tif'
    win.save_result(dest)
    pump(app, lambda: win.export_worker is None)
    with tifffile.TiffFile(dest) as tf:
        assert len(tf.pages) == 5
        assert json.loads(tf.pages[0].description)['result']['reference_epoch'] == '0.0'


def test_gui_bayer_override_and_per_channel_validity(gui, tmp_path):
    app, win = gui
    win.path = write_ser(tmp_path/'bayer.ser', np.full((2,9,12), 40, np.uint16))
    win.controls.fields['bayer_override'].setCurrentText('BGGR')
    win._run()
    pump(app, lambda: win.job is None)
    r = win.last_result
    assert r.channel_order == 'RGB' and r.validity.shape == (9,12,3)
    assert np.all(r.validity.sum(axis=2) == 1)
    win.view.setCurrentText('Validity')
    win.channel.setCurrentText('B')
    assert win._view_data()[0].shape == r.validity.shape


@pytest.mark.hardware
def test_gui_heartbeat_under_full_cpu_load(gui, tmp_path):
    """Bounded local responsiveness check; opt in with --run-hardware."""
    import subprocess
    import sys
    from PySide6.QtCore import QTimer
    app, win = gui
    frame = np.arange(128*128, dtype=np.uint16).reshape(128,128)
    win.path = write_ser(tmp_path/'load.ser', np.stack([frame]*200))
    workers, ticks = [], []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    try:
        script = 'import time\nend=time.monotonic()+3\nx=1\nwhile time.monotonic()<end: x=(x*31+1)%99991'
        for _ in range(min(32, os.cpu_count() or 1)):
            workers.append(subprocess.Popen([sys.executable, '-c', script]))
        win._run()
        timer.start()
        started = time.monotonic()
        pump(app, lambda: time.monotonic()-started > 2 and win.job is None, timeout=15)
        assert len(ticks) >= 10
        assert max(np.diff(ticks)) < 1.0
        print(f'CPU-load GUI heartbeat: {len(workers)} burners, maximum gap {max(np.diff(ticks)):.3f}s')
    finally:
        timer.stop()
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
            worker.wait(timeout=5)
