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
@pytest.mark.parametrize('cached', [False, True])
def test_selecting_capture_stays_idle_until_requested(monkeypatch, tmp_path, entry, cached):
    from planetrecon.gui.app import MainWindow, create_app
    from planetrecon.gui import settings
    from planetrecon.reconstruction import ReconstructionConfig
    app = create_app([])
    capture = write_ser(tmp_path/'input.ser', np.full((16, 48, 48), 100, dtype='u2'))
    cache = capture.with_name(capture.name+'.planetrecon-preprocess.npz')
    if cached:
        # Even an unusable cache must be left alone until explicit inspection/run.
        cache.write_bytes(b'not yet validated')
    def unexpected_job(*args, **kwargs):
        pytest.fail('Selecting a capture must not start a worker')
    monkeypatch.setattr('planetrecon.gui.app.start_stack_job', unexpected_job)
    cfg = ReconstructionConfig(device='cpu', threads=2)
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
        for _ in range(5):
            app.processEvents()
        assert win.path == capture
        assert win.job is None and win.run_stage is None
        assert win.input_image is None
        assert win.run_btn.isEnabled()
        assert win.preprocessing_info['status'] == ('unverified' if cached else 'missing')
        assert 'Run' in win.status.text()
        assert cache.exists() == cached
        if cached:
            assert cache.read_bytes() == b'not yet validated'
    finally:
        win.window.close()
        app.processEvents()
