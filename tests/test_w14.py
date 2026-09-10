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
from planetrecon.export import ExportCancelled, ExportConfig, export_result, parse_tiff_description
from planetrecon.io.ser import write_ser
from planetrecon.io.source import ArraySource
from planetrecon.jobs import JobEvent, JobHandle, result_payload, start_stack_job
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult


def config(**kw):
    return ReconstructionConfig(frame_preselection=False, device='cpu', threads=2, batch_frames=1, **kw)


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


def test_fit_preview_tracks_canvas_resize_from_result_metadata(gui):
    app, win = gui
    win._accept_result(result_payload(result(incomplete=False)))
    pump(app, lambda: win.image_label.pixmap() is not None)
    # A taller metadata area changes the preview height without changing the
    # top-level window size, as happens with longer result/source descriptions.
    win.result_label.setMinimumHeight(150)
    for _ in range(8):
        app.processEvents()
    pixmap = win.image_label.pixmap()
    assert pixmap.height() <= win.canvas.viewport().height()
    assert pixmap.width() <= win.canvas.viewport().width()


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
        meta = parse_tiff_description(tf.pages[0].description)
    assert image.shape == (514,12,3)
    np.testing.assert_allclose(image * meta['mapping']['white'], 3., rtol=1e-7)
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
        assert len(tf.pages) == 1
        meta = parse_tiff_description(tf.pages[0].description)
        assert meta['result']['reference_epoch'] == '0.0'
        assert not tf.pages[0].is_shaped
    coverage = dest.with_name(meta['coverage_file'])
    with tifffile.TiffFile(coverage) as tf:
        roles = [json.loads(page.description)['role'] for page in tf.pages]
        assert roles[:2] == ['validity', 'coverage']
        assert {json.loads(page.description)['layer'] for page in tf.pages[2:]} == {'globe', 'ring'}


def test_gui_bayer_override_and_per_channel_validity(gui, tmp_path):
    app, win = gui
    win.path = write_ser(tmp_path/'bayer.ser', np.full((2,9,12), 40, np.uint16))
    win.controls.fields['bayer_override'].setCurrentText('BGGR')
    win._run()
    pump(app, lambda: win.job is None)
    r = win.last_result
    assert r.channel_order == 'RGB' and r.validity.shape == (9,12,3)
    assert r.validity.all()
    np.testing.assert_allclose(r.image, 40.)
    direct = np.stack([r.layer_coverage[f"cfa_direct_{c}"] for c in "RGB"], axis=2)
    assert np.all((direct > 0).sum(axis=2) == 1)
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


def test_new_runs_snapshot_every_processing_control(gui, tmp_path, monkeypatch):
    """All visible settings, including cleared values, reach each fresh job."""
    import planetrecon.gui.app as module
    from PySide6.QtWidgets import QComboBox, QCheckBox, QSpinBox
    _, win = gui
    calls = []
    def start(path, cfg, **options):
        calls.append((path, cfg, options))
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None)
    monkeypatch.setattr(module, 'start_stack_job', start)
    win.path = tmp_path/'input.ser'
    win.device.setCurrentText('auto')
    win._run()
    first = calls[-1][1]
    win._finish_job()
    edits = dict(device='gpu', threads=7, batch_frames=3, max_ram_bytes=None,
        max_vram_bytes=None, crop='bland', bayer_override='BGGR', endian_override='big',
        endian_convention='spec', recover_complete_frames=True, reject_saturated=False, frame_preselection=True,
        stack_percent=25, frame_selection_mode='frame_count', local_alignment=False, local_patch_size=33, squared_quality_weights=False, reference_index=2, max_shift_px=9.5, cadence_s=.2, exposure_s=.1,
        bias_path='bias.npy', dark_path='dark.npy', flat_path='flat.npy',
        gain_e_per_adu=2.5, read_noise_e=3., saturate_adu=4000., geometry_mode='saturn',
        reference_epoch_s=1., field_angle0_rad=5., field_rate_rad_s=2., surface_rate_rad_s=3.,
        rotation_planet='saturn', reverse_rotation=True,
        field_center_x=32., field_center_y=24., equatorial_radius_px=12., flattening=.1,
        pole_pa_rad=10., sub_obs_lat_rad=20., sub_obs_lon0_rad=30.,
        ring_inner_radius_px=16., ring_outer_radius_px=26., ring_transmission=.5,
        sun_lon_rad=40., sun_lat_rad=50., moon_x=10., moon_y=12., moon_radius_px=2.,
        moon_vx_px_s=.3, moon_vy_px_s=.4)
    assert set(edits) == set(win.controls.fields), 'New controls need restart coverage'
    expected = {}
    for key, value in edits.items():
        edit = win.controls.fields[key]
        if isinstance(edit, QComboBox):edit.setCurrentIndex(edit.findData(value))
        elif isinstance(edit, QCheckBox):edit.setChecked(value)
        elif isinstance(edit, QSpinBox):
            edit.setKeyboardTracking(False)
            edit.lineEdit().setText(str(value))  # Uncommitted keyboard input.
        else:edit.setText('' if value is None else str(value))
        if value is not None and key in win.controls.angular:value = np.radians(value)
        if value is not None and key in ('max_ram_bytes', 'max_vram_bytes'):value = int(value*1024**2)
        expected[key] = value
    win.checkpoint_path.setText(str(tmp_path/'new-state.npz'))
    win.resume_check.setChecked(True)
    win._run()
    second = calls[-1][1]
    assert second == replace(first, **expected)
    assert first.device == 'auto' and first.geometry_mode == 'none'
    assert calls[-1][2] == dict(state_checkpoint=tmp_path/'new-state.npz', resume_from=tmp_path/'new-state.npz')
    win._finish_job()
    # Clearing optional fields must not resurrect values from an earlier run.
    for key in ('bias_path','dark_path','flat_path','max_vram_bytes','gain_e_per_adu'):
        win.controls.fields[key].clear()
        expected[key] = None
    win.device.setCurrentText('cpu');expected['device'] = 'cpu'
    win.controls.fields['max_ram_bytes'].setText('1024');expected['max_ram_bytes'] = 1024**3
    win.checkpoint_path.clear();win.resume_check.setChecked(False)
    win._run()
    assert calls[-1][1] == replace(second, **expected) and calls[-1][2] == {}
    win._finish_job()
    win.device.setCurrentText('gpu')
    win.controls.fields['geometry_mode'].setCurrentText('none')
    win.controls.fields['max_ram_bytes'].clear()
    win.controls.fields['max_vram_bytes'].setText('64.5')
    win._run()
    assert calls[-1][1].max_vram_bytes == int(64.5*1024**2)
    assert calls[-1][1].max_ram_bytes is None and calls[-1][1].ring_inner_radius_px is None


def test_cancel_restart_uses_new_settings_and_labels_previous_result(gui, tmp_path):
    app, win = gui
    frame = np.arange(96, dtype='u2').reshape(8,12)+100
    win.path = write_ser(tmp_path/'rerun.ser', np.stack([frame]*8))
    old = result(incomplete=False)
    win._accept_result(result_payload(old))
    win._run();win._cancel()
    pump(app, lambda: win.job is None, timeout=5)
    f = win.controls.fields
    f['bayer_override'].setCurrentText('RGGB')
    f['gain_e_per_adu'].setText('2')
    f['reference_index'].setValue(2)
    f['batch_frames'].setValue(3)
    win._run()
    assert 'Previous result' in win.result_label.text()
    assert 'backend pending' in win.run_device.text()
    assert 'example warning' not in win.warnings.text()
    assert json.loads(win.details.toPlainText())['current_run_config']['gain_e_per_adu'] == 2
    pump(app, lambda: win.job is None)
    r = win.last_result
    assert not win.error.text() and r is not old and r.channel_order == 'RGB'
    assert r.provenance['config']['reference_index'] == 2
    assert r.provenance['config']['batch_frames'] == 3
    direct = np.stack([r.layer_coverage[f"cfa_direct_{c}"] > 0 for c in "RGB"], axis=2)
    np.testing.assert_allclose((r.image * direct).sum(axis=2), frame*2, rtol=1e-12, atol=1e-10)
    assert r.validity.all()
    assert 'Previous result' not in win.result_label.text()
    assert 'requested CPU' in win.run_device.text() and 'using CPU' in win.run_device.text()


def test_current_backend_and_fallback_are_shown_without_valid_image(gui):
    _, win = gui
    win.config = replace(win.config, device='gpu')
    payload = result_payload(result())
    payload['n_used'] = 0
    payload['provenance']['device_report'] = {'reason': 'cuda_not_available', 'fallback': True}
    payload['warnings'] = ['GPU unavailable; continuing on CPU']
    win._accept_result(payload)
    assert win.last_result is None
    assert 'requested GPU' in win.run_device.text() and 'using CPU' in win.run_device.text()
    assert 'cuda_not_available' in win.run_device.text()
    assert 'GPU unavailable' in win.warnings.text()


@pytest.mark.hardware
def test_cancel_auto_then_gpu_then_cpu_in_same_gui(gui, tmp_path):
    if os.environ.get('PLANETRECON_TEST_GPU') != '1':pytest.skip('explicit GPU opt-in required')
    app, win = gui
    frame = np.arange(96, dtype='u2').reshape(8,12)+100
    win.path = write_ser(tmp_path/'device-rerun.ser', np.stack([frame]*8))
    win.device.setCurrentText('auto')
    win._run();win._cancel()
    pump(app, lambda: win.job is None, timeout=6)
    for device, backend, gain in [('gpu','cuda',2), ('cpu','cpu',3)]:
        win.device.setCurrentText(device)
        win.controls.fields['gain_e_per_adu'].setText(str(gain))
        win._run()
        assert win.config.device == device and 'backend pending' in win.run_device.text()
        pump(app, lambda: win.job is None, timeout=15)
        assert not win.error.text()
        r = win.last_result
        assert r.backend == backend, r.provenance.get('device_report')
        assert r.provenance['config']['device'] == device
        np.testing.assert_allclose(r.image, frame*gain, rtol=1e-12, atol=1e-10)
        assert f'using {backend.upper()}' in win.run_device.text()


def test_save_dialog_reads_changed_encoding_and_mapping_each_time(gui, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    _, win = gui
    win._accept_result(result_payload(result()))
    saved = []
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(tmp_path/'output'), ''))
    monkeypatch.setattr(win, 'save_result', lambda path, cfg, **kw: saved.append((path,cfg)))
    for encoding, black, white, gamma in [('png16','2','200','2.2'), ('tiff16','4','400','1.2')]:
        win.encoding.setCurrentText(encoding)
        win.save_black.setText(black);win.save_white.setText(white);win.save_gamma.setText(gamma)
        win._choose_save()
        assert saved[-1][1] == ExportConfig(encoding,float(black),float(white),float(gamma))
    win.encoding.setCurrentText('tiff32');win._choose_save()
    assert saved[-1][1] == ExportConfig('tiff32', 4., 400.)
    assert win.save_black.isEnabled() and win.save_white.isEnabled() and not win.save_gamma.isEnabled()
    win.encoding.setCurrentText('tiff32_raw');win._choose_save()
    assert saved[-1][1] == ExportConfig('tiff32_raw')
    assert not win.save_black.isEnabled() and not win.save_white.isEnabled() and not win.save_gamma.isEnabled()
    assert saved[0][0].suffix == '.png' and saved[-1][0].suffix == '.tif'
    saved.clear()
    win.black.setValue(10); win.white.setValue(50)
    win.save_black.clear(); win.save_white.clear(); win.save_gamma.clear()
    for encoding in ('tiff32', 'tiff16', 'png16'):
        win.encoding.setCurrentText(encoding)
        win._choose_save()
        assert saved[-1][1] == ExportConfig(encoding, 10.0, 50.0)
    win.save_black.setText('not-a-number')
    win._choose_save()
    assert 'not a number' in win.save_status.text()
    assert saved[-1][1] == ExportConfig('png16', 10.0, 50.0)


def test_jupiter_sized_green_result_display_does_not_lose_bayer_support(gui):
    _, win = gui
    y,x = np.indices((424,656))
    green = (x+y)%2 == 1
    valid = np.stack([~green,green,~green],axis=2)
    r = ReconstructionResult(np.broadcast_to([10.,20.,30.],valid.shape).copy(),
        valid.astype(float),valid,'adu','RGB','cpu','float64','final',False,n_used=1,
        provenance={'color_mode':'RGGB'})
    win._accept_result(result_payload(r))
    win.channel.setCurrentText('G')
    win._draw()
    assert win.preview.validity[...,1].all()
    assert 'valid 100.0%' in win.histogram.text()
    assert win.last_result.validity[...,1].mean() == .5
    np.testing.assert_array_equal(win.last_result.validity,r.validity)
    assert win.image_label.pixmap().toImage().pixelColor(0,0).getRgb()[:3] != (180,40,160)


@pytest.mark.parametrize('bits,peak,gain,expected', [
    (8,100.,None,143.), (8,250.,None,255.),
    (16,100.,None,143.), (8,200.,2.,286.), (8,1000.,2.,510.),
])
def test_white_fit_uses_full_result_peak_or_capture_range(gui,bits,peak,gain,expected):
    _,win = gui
    r = result(value=10)
    r.image[1,1,0] = peak  # Omitted by this 514-row result's stride-2 preview.
    r.image[3,1,1] = 1e9
    r.validity[3,1,1] = False  # Unsupported samples must not affect fitting.
    r.provenance = {'source':{'bit_depth':bits,'units':'adu'},
                    'config':{'gain_e_per_adu':gain}}
    r.units = 'e-' if gain else 'adu'
    win._accept_result(result_payload(r))
    assert win.white.value() == pytest.approx(expected)
    assert win.white.value() <= ((1 << bits)-1)*(gain or 1.)
    win.white.setValue(42.)  # Later results still preserve explicit user levels.
    win._accept_result(result_payload(r))
    assert win.white.value() == 42.
    win._fit_levels()
    assert win.white.value() == pytest.approx(expected)
    win.view.setCurrentText('Validity');win._fit_levels()
    assert win.white.value() == pytest.approx(1.)


def test_input_white_fit_uses_raw_peak_and_bit_depth(gui):
    _,win = gui
    win.inspecting = True
    win._set_input({'input_image':np.full((8,12),20,dtype='u2'), 'input_max':65535.,
        'input_view':'mono','source_metadata':{'path':'input.ser','width':24,'height':16,
        'n_frames':1,'color_mode':'mono','bit_depth':16,'units':'adu'}})
    assert win.white.value() == pytest.approx(65535)


def test_uniform_positive_channel_coverage_remains_visible(gui):
    _, win = gui
    r = result(value=10)
    r.coverage.fill(104.)
    win._accept_result(result_payload(r))
    win.view.setCurrentText('Coverage')
    win.channel.setCurrentText('R')
    win._fit_levels()
    assert win.black.value() == 0.
    assert win.white.value() == pytest.approx(1.43*104.)
    pixel = win.image_label.pixmap().toImage().pixelColor(0, 0)
    assert pixel.red() > 100 and pixel.red() == pixel.green() == pixel.blue()


def test_qimage_automatic_white_does_not_clip_sparse_bright_pixel():
    from planetrecon.gui.app import _to_qimage
    image = np.full((20,20),10,dtype='u1');image[0,0] = 200
    rendered = _to_qimage(image)
    assert rendered.pixelColor(0,0).red() < 255


def test_inspection_white_includes_bright_pixel_missing_from_preview(gui,tmp_path):
    app,win = gui
    raw = np.full((8,1026),10,dtype='u2')
    raw[1,1] = 15000
    win.path = write_ser(tmp_path/'bright.ser',raw[None])
    win._inspect();pump(app,lambda:win.job is None)
    assert win.input_image.max() == 10
    assert win.white.value() == pytest.approx(21450)


def test_new_runs_apply_stronger_quality_weight_toggle(gui, tmp_path, monkeypatch):
    import planetrecon.gui.app as module
    _, win = gui
    configs = []
    def start(path, cfg, **options):
        configs.append(cfg)
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None)
    monkeypatch.setattr(module, 'start_stack_job', start)
    win.path = tmp_path/'input.ser'
    win.controls.fields['frame_preselection'].setChecked(True)
    win.controls.fields['local_alignment'].setChecked(True)
    stronger = win.controls.fields['squared_quality_weights']
    for enabled in (False, True, False):
        stronger.setChecked(enabled)
        win._run()
        assert configs[-1].local_alignment
        assert configs[-1].squared_quality_weights is enabled
        win._finish_job()


def test_saturn_run_lists_missing_geometry_before_starting_a_worker(gui, tmp_path, monkeypatch):
    import planetrecon.gui.app as module
    _, win = gui
    calls = []
    def start(path, cfg, **options):
        calls.append((cfg, options))
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None)
    monkeypatch.setattr(module, 'start_stack_job', start)
    win.path = tmp_path/'saturn.ser'
    fields = win.controls.fields
    fields['geometry_mode'].setCurrentText('saturn')
    win._run()
    assert not calls and win.job is None
    assert win.controls.currentIndex() == 2
    assert 'Signed observer latitude' not in win.error.text()
    assert 'Globe equatorial radius' in win.error.text()
    assert 'Inner ring radius' in win.error.text() and 'Outer ring radius' in win.error.text()
    assert 'Motion model None' in win.error.text()
    assert 'sub_obs_lat_rad' not in win.error.text()
    # Discovery and inspection remain usable with incomplete physical geometry.
    win._preprocess()
    assert calls[-1][1] == {'preprocess_only': True}
    win._finish_job()
    win._inspect()
    assert calls[-1][1] == {'inspect_only': True}
    win._finish_job()
    for key, value in {'sub_obs_lat_rad': '-12', 'equatorial_radius_px': '30',
                       'ring_inner_radius_px': '40', 'ring_outer_radius_px': '60',
                       'surface_rate_rad_s': '0'}.items():
        fields[key].setText(value)
    win._run()
    assert calls[-1][1] == {} and calls[-1][0].sub_obs_lat_rad == pytest.approx(np.radians(-12))
    win._finish_job()
    # Ordinary Saturn stacking never needs these physical parameters.
    fields['geometry_mode'].setCurrentText('none')
    fields['sub_obs_lat_rad'].clear()
    fields['equatorial_radius_px'].clear()
    win._run()
    assert calls[-1][0].geometry_mode == 'none'
    win._finish_job()


def test_surface_run_requires_rate_but_preprocessing_remains_available(gui, tmp_path, monkeypatch):
    import planetrecon.gui.app as module
    _, win = gui
    calls = []
    def start(path, cfg, **options):
        calls.append((cfg, options))
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None)
    monkeypatch.setattr(module, 'start_stack_job', start)
    win.path = tmp_path/'input.ser'
    fields = win.controls.fields
    fields['geometry_mode'].setCurrentText('surface')
    win._run()
    assert not calls and win.job is None
    assert 'Surface rotation rate' in win.error.text()
    assert win.controls.currentIndex() == 2
    win._preprocess()
    assert calls[-1][1] == {'preprocess_only': True}
    win._finish_job()
    fields['surface_rate_rad_s'].setText('0')
    win._run()
    assert calls[-1][1] == {} and calls[-1][0].surface_rate_rad_s == 0.
    win._finish_job()
