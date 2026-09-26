"""Run validates/rebuilds preprocessing before using current GUI settings."""
import numpy as np
import pytest
from planetrecon.io.ser import write_ser
from test_w14 import gui, pump


def test_run_builds_cache_reuses_it_and_rebuilds_after_input_change(gui, tmp_path):
    app, win = gui
    y, x = np.indices((48, 48))
    frame = (1000*np.exp(-((x-24)**2+(y-24)**2)/90)*(1+.1*np.cos(x))).astype('u2')
    win.path = write_ser(tmp_path / 'input.ser', np.repeat(frame[None], 16, axis=0))
    win.controls.fields['cadence_s'].setText('0.1')
    win._run()
    pump(app, lambda: win.job is None, timeout=20)
    assert not win.error.text()
    assert win.last_result.n_used == 16
    info = win.preprocessing_info
    assert info['status'] == 'ready'
    from pathlib import Path
    cache = Path(info['path'])
    stamp = cache.stat().st_mtime_ns
    assert win.controls.configuration().reference_epoch_s == .75
    win._run()
    pump(app, lambda: win.job is None, timeout=20)
    assert not win.error.text()
    assert cache.stat().st_mtime_ns == stamp
    win.controls.fields['gain_e_per_adu'].setText('2')
    win._run()
    pump(app, lambda: win.job is None, timeout=20)
    assert not win.error.text()
    assert cache.stat().st_mtime_ns != stamp
    assert win.last_result.units == 'e-'


def test_cancel_automatic_preparation_does_not_start_stack(gui, tmp_path):
    app, win = gui
    win.path = write_ser(tmp_path / 'input.ser', np.full((16, 48, 48), 100, dtype='u2'))
    win._run()
    assert win.run_stage == 'inspect'
    win._cancel()
    pump(app, lambda: win.job is None)
    assert win.run_stage is None
    assert win.last_result is None
    assert win.run_btn.isEnabled()


@pytest.mark.parametrize('entry', ['startup', 'restored', 'open'])
@pytest.mark.parametrize('cached', ['missing', 'valid', 'invalid'])
def test_loading_capture_automatically_displays_cached_quality(monkeypatch, tmp_path, entry, cached):
    from planetrecon.gui.app import MainWindow, create_app
    from planetrecon.gui import app as module, settings
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.preprocess_cache import preprocess_source
    from planetrecon.reconstruction import ReconstructionConfig
    app = create_app([])
    y, x = np.indices((48, 48))
    frame = (1000*np.exp(-((x-24)**2+(y-24)**2)/90)*(1+.1*np.cos(x))).astype('u2')
    capture = write_ser(tmp_path/'input.ser', np.repeat(frame[None], 8, axis=0))
    cache = capture.with_name(capture.name+'.planetrecon-preprocess.npz')
    cfg = ReconstructionConfig(device='cpu', threads=2)
    if cached == 'valid':
        with SERSource(capture) as source:
            preprocess_source(source, cfg)
    elif cached == 'invalid':
        cache.write_bytes(b'not a cache')
    stamp = cache.stat().st_mtime_ns if cache.exists() else None
    calls = []
    start_job = module.start_stack_job
    def start(*args, **options):
        calls.append(options)
        return start_job(*args, **options)
    monkeypatch.setattr(module, 'start_stack_job', start)
    preferences = tmp_path/'settings.json'
    if entry == 'restored':
        saved = MainWindow(config=cfg)
        settings.save(preferences, {'capture': str(capture), 'controls': saved.controls.settings_state()})
        saved.window.close()
        win = MainWindow(settings_path=preferences)
    else:
        win = MainWindow(capture if entry == 'startup' else None, cfg)
    try:
        if entry == 'open':
            win.input_image = np.ones((12, 12))
            monkeypatch.setattr('planetrecon.gui.app.QFileDialog.getOpenFileName',
                                lambda *args: (str(capture), 'Captures'))
            win.open_btn.click()
        win.show()
        if entry == 'restored':
            for _ in range(5):
                app.processEvents()
            assert win.path is None and win.job is None and win.frame_job is None
            assert not calls and win.input_image is None
            assert win.quality_plot.selection is None
            assert not win.cancel_btn.isEnabled() and not win.run_btn.isEnabled()
            assert win.open_btn.isEnabled()
            if cache.exists():
                assert cache.stat().st_mtime_ns == stamp
            return
        if cached != 'missing':
            assert win.job is not None and win.cancel_btn.isEnabled()
        pump(app, lambda: win.job is None, timeout=20)
        assert win.path == capture and win.run_stage is None
        assert win.last_result is None and win.run_btn.isEnabled()
        assert not win.cancel_btn.isEnabled()
        assert not hasattr(win, 'inspect_btn')
        assert win.preprocessing_info['status'] == {'valid': 'ready', 'invalid': 'invalid', 'missing': 'missing'}[cached]
        assert cache.exists() == (cached != 'missing')
        assert len(calls) == int(cached != 'missing')
        if cached != 'missing':
            assert calls[0]['inspect_only']
            assert win.input_image is not None
            assert cache.stat().st_mtime_ns == stamp
        if cached == 'valid':
            assert win.quality_plot.selection is not None
            assert win.preview_tabs.currentWidget() is win.quality_plot
        else:
            assert win.quality_plot.selection is None
    finally:
        win.window.close()
        app.processEvents()


def test_cancel_automatic_cache_loading_returns_idle(gui, tmp_path):
    app, win = gui
    win.path = write_ser(tmp_path/'input.ser', np.full((8, 48, 48), 100, dtype='u2'))
    win.path.with_name(win.path.name+'.planetrecon-preprocess.npz').write_bytes(b'bad cache')
    win._capture_ready()
    assert win.cancel_btn.isEnabled()
    win._cancel()
    pump(app, lambda: win.job is None)
    assert win.run_stage is None and not win.cancel_btn.isEnabled()
    assert win.run_btn.isEnabled() and win.last_result is None


@pytest.mark.parametrize('event_kind', ['source', 'completed'])
def test_result_display_error_releases_finished_worker(gui, monkeypatch, event_kind):
    from types import SimpleNamespace
    from planetrecon.jobs import JobEvent
    _, win = gui
    closed = []
    events = [JobEvent('test', 1, event_kind, {'source_metadata': {}})]
    if event_kind == 'source':
        events.append(JobEvent('test', 2, 'completed', {'source_metadata': {}}))
    win.job = SimpleNamespace(job_id='test', poll=lambda: events, close=lambda: closed.append(True))
    win.run_stage = 'inspect'
    win._buttons()
    monkeypatch.setattr(win, '_set_input', lambda _: (_ for _ in ()).throw(ValueError('bad preview')))
    win._poll()
    assert closed == [True] and win.job is None and win.run_stage is None
    assert not win.cancel_btn.isEnabled()
    assert 'bad preview' in win.error.text()
