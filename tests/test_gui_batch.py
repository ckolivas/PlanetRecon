"""Real sequential batch jobs, exports, isolation and cancellation."""
import numpy as np
import tifffile

from planetrecon.gui.batch import make_queue
from planetrecon.io.ser import write_ser
from planetrecon.export import parse_tiff_description
from test_w14 import gui, pump


def small_frames(win):
    # Batch stacks like Stack (alignment points on); these 48 px captures need small points.
    win.controls.fields['ap_size'].setValue(24)


def capture(path, scale=1000, n=8):
    y, x = np.indices((48, 48))
    frame = (scale*np.exp(-((x-24)**2+(y-24)**2)/90)*(1+.1*np.cos(x))).astype('u2')
    return write_ser(path, np.repeat(frame[None], n, axis=0))


def test_batch_outputs_each_capture_and_continues_after_failure(gui, tmp_path):
    app, win = gui
    first = capture(tmp_path / 'first.ser')
    second = capture(tmp_path / 'second.ser', scale=2000, n=12)
    output = tmp_path / 'output'
    output.mkdir()
    win.controls.fields['cadence_s'].setText('0.1')
    win.controls.fields['gain_e_per_adu'].setText('2')
    small_frames(win)
    assert win.start_batch([first, tmp_path / 'missing.ser', second], output)
    assert not win.controls.isEnabled() and not win.encoding.isEnabled()
    assert not win.batch_btn.isEnabled() and win.cancel_btn.isEnabled()
    # Run is retired; Batch stacks each capture as Stack does.
    assert not hasattr(win, 'run_btn')
    pump(app, lambda: not win.batch_active and win.export_worker is None, timeout=30)
    assert [i.status for i in win.batch_items] == ['saved', 'failed', 'saved']
    assert 'missing.ser' in win.batch_items[1].detail
    # Each completed capture lists its stack quality beneath it in the queue.
    assert 'stacks 8 of 8 screened frames (Fourier power 4 weights' in win.batch_items[0].quality
    assert 'stacks 12 of 12 screened frames' in win.batch_items[2].quality
    assert not win.batch_items[1].quality
    assert win.batch_status.toPlainText().count('Stack quality: ') == 2
    # Output names carry frames, weighted quality and estimated SNR.
    assert win.batch_items[0].output.name.startswith('first-planetrecon-n8-q')
    assert '-snr' in win.batch_items[2].output.name and win.batch_items[2].output.suffix == '.tif'
    assert win.batch_btn.isEnabled() and win.controls.isEnabled()
    for item, count, epoch in ((win.batch_items[0], 8, .35), (win.batch_items[2], 12, .55)):
        with tifffile.TiffFile(item.output) as image:
            metadata = parse_tiff_description(image.pages[0].description)
            assert metadata['result']['n_used'] == count
            assert metadata['result']['provenance']['config']['reference_epoch_s'] == epoch
            assert metadata['result']['units'] == 'e-'
            assert metadata['result']['provenance']['source']['path'] == str(item.source)
            assert not metadata['result']['incomplete']
            assert np.isfinite(image.asarray()).all()
    # Blank export levels are fitted separately, not inherited from the first.
    assert '2 saved, 1 failed' in win.status.text()
    assert win.last_result.n_used == 12


def test_quality_tag_never_replaces_existing_or_reserved_outputs(tmp_path):
    from planetrecon.gui.batch import quality_tag, with_quality
    summary = {'frames_stacked': 3341, 'quality': {'weighted_mean': 1.19267}, 'estimated_snr': 1014.96,
               'alignment_points': {'frames_per_point': {'median': 2105.}, 'weighted_quality': {'median': 1.21546}}}
    assert quality_tag(summary) == '-n2105-q1.215-snr1015'
    assert quality_tag({**summary, 'alignment_points': None}) == '-n3341-q1.193-snr1015'
    assert quality_tag({'unavailable': 'old checkpoint'}) == ''
    output = tmp_path / 'jup-planetrecon.tif'
    (tmp_path / 'jup-planetrecon-n2105-q1.215-snr1015.tif').write_bytes(b'keep')
    reserved = {tmp_path / 'jup-planetrecon-n2105-q1.215-snr1015-2.tif'}
    assert with_quality(output, summary, reserved).name == 'jup-planetrecon-n2105-q1.215-snr1015-3.tif'
    assert with_quality(output, None) == output


def test_queue_duplicate_names_and_existing_outputs(tmp_path):
    existing = tmp_path / 'same-planetrecon.tif'
    existing.write_bytes(b'keep')
    paths = [tmp_path / 'a/same.ser', tmp_path / 'b/same.ser']
    queue = make_queue([*paths, paths[0]], tmp_path, 'tiff32')
    assert [i.output.name for i in queue] == ['same-planetrecon-2.tif', 'same-planetrecon-3.tif']
    assert existing.read_bytes() == b'keep'


def test_cancel_stops_pending_batch_files(gui, tmp_path):
    app, win = gui
    first = capture(tmp_path / 'first.ser')
    second = capture(tmp_path / 'second.ser')
    small_frames(win)
    assert win.start_batch([first, second], tmp_path)
    win._cancel()
    pump(app, lambda: win.job is None)
    assert not win.batch_active
    assert [i.status for i in win.batch_items] == ['cancelled', 'cancelled']
    assert not any(i.output.exists() for i in win.batch_items)


def test_batch_clears_automatic_values_but_retains_manual_overrides(gui, tmp_path, monkeypatch):
    _, win = gui
    import threading
    from types import SimpleNamespace
    calls = []
    def start(path, config, **kwargs):
        calls.append(config)
        return SimpleNamespace(snapshot_request=threading.Event(), close=lambda: None)
    monkeypatch.setattr('planetrecon.gui.app.start_stack_job', start)
    win.controls.fields['equatorial_radius_px'].setText('90')
    win.controls.geometry_auto['equatorial_radius_px'] = '90'
    win.controls.flip_pole_button.click()
    win.controls.fields['reference_epoch_s'].setText('45')
    win.controls.geometry_auto['reference_epoch_s'] = '45'
    win.controls.suggest_capture_planet('old-Jup.ser')
    small_frames(win)
    assert win.start_batch([tmp_path / 'Sat.ser', tmp_path / 'Mars.ser'], tmp_path)
    assert calls[-1].equatorial_radius_px is None
    assert calls[-1].reference_epoch_s == 0
    assert calls[-1].pole_pa_rad == np.pi
    assert win.controls.flip_pole_button.isChecked()
    assert calls[-1].rotation_planet == 'saturn'
    win._finish_job()
    win.batch_current.status = 'saved'
    win._next_batch()
    assert calls[-1].rotation_planet == 'mars'
    assert win.controls.flip_pole_button.isChecked()
    assert calls[-1].equatorial_radius_px is None
    win._shutdown()


def test_batch_export_collision_fails_without_overwriting_and_continues(gui, tmp_path, monkeypatch):
    import planetrecon.gui.app as module
    app, win = gui
    first = capture(tmp_path / 'first.ser')
    second = capture(tmp_path / 'second.ser')
    small_frames(win)
    original = module.export_result
    raced = []
    def racing_export(result, path, *args, **kwargs):
        if not raced:
            # Another program creates the chosen destination just before publication.
            path.write_bytes(b'external image')
            raced.append(path)
        return original(result, path, *args, **kwargs)
    monkeypatch.setattr(module, 'export_result', racing_export)
    assert win.start_batch([first, second], tmp_path)
    pump(app, lambda: not win.batch_active and win.export_worker is None, timeout=30)
    assert [i.status for i in win.batch_items] == ['failed', 'saved']
    assert win.batch_items[0].output.read_bytes() == b'external image'
    assert win.batch_items[0].detail


def test_cancel_during_batch_export_stops_the_queue(gui, tmp_path, monkeypatch):
    import threading
    import planetrecon.gui.app as module
    app, win = gui
    entered, release = threading.Event(), threading.Event()
    original = module.export_result
    def blocked_export(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'export_result', blocked_export)
    paths = [capture(tmp_path / 'first.ser'), capture(tmp_path / 'second.ser')]
    try:
        small_frames(win)
        assert win.start_batch(paths, tmp_path)
        pump(app, entered.is_set, timeout=20)
        assert win.batch_saving and win.batch_items[0].status == 'saving'
        win._cancel()
        release.set()
        pump(app, lambda: win.export_worker is None)
        assert not win.batch_active and win.job is None
        assert [i.status for i in win.batch_items] == ['cancelled', 'cancelled']
        assert not any(i.output.exists() for i in win.batch_items)
    finally:
        release.set()


def faulting_starts(monkeypatch, faults):
    """The first ``faults`` stack workers fail with a CUDA launch timeout."""
    import threading
    from types import SimpleNamespace
    import planetrecon.gui.app as module
    from planetrecon.jobs import JobEvent
    real, stacks = module.start_stack_job, []

    def start(*args, **kwargs):
        if kwargs.get('inspect_only') or kwargs.get('preprocess_only'):
            return real(*args, **kwargs)
        stacks.append(kwargs)
        if len(stacks) > faults:
            return real(*args, **kwargs)
        message = ('CUDA error: the launch timed out and was terminated\n'
                   "Search for `cudaErrorLaunchTimeout' in the CUDA runtime documentation.")
        return SimpleNamespace(job_id='fault', snapshot_request=threading.Event(), cancel_event=threading.Event(),
                               process=SimpleNamespace(is_alive=lambda: False), close=lambda: None,
                               poll=lambda timeout=0.: [JobEvent('fault', 1, 'error', {'message': message})])
    monkeypatch.setattr(module, 'start_stack_job', start)
    return stacks


def test_gpu_fault_restarts_the_stack_once_in_a_new_worker(gui, tmp_path, monkeypatch):
    app, win = gui
    stacks = faulting_starts(monkeypatch, faults=1)
    small_frames(win)
    assert win.start_batch([capture(tmp_path / 'first.ser')], tmp_path)
    pump(app, lambda: not win.batch_active and win.export_worker is None, timeout=30)
    item = win.batch_items[0]
    assert item.status == 'saved' and len(stacks) == 2
    assert 'recovered from a GPU fault (CUDA error: the launch timed out' in item.detail
    assert 'Recovered from a GPU fault' in win.warnings.text() and not win.error.text()


def test_repeated_gpu_fault_fails_the_capture(gui, tmp_path, monkeypatch):
    app, win = gui
    stacks = faulting_starts(monkeypatch, faults=2)
    small_frames(win)
    assert win.start_batch([capture(tmp_path / 'first.ser'), capture(tmp_path / 'second.ser')], tmp_path)
    pump(app, lambda: not win.batch_active and win.export_worker is None, timeout=30)
    # One restart per capture: the first fails after two faults, the second retries afresh.
    assert [i.status for i in win.batch_items] == ['failed', 'saved'] and len(stacks) == 3
    assert 'launch timed out' in win.batch_items[0].detail
