"""Run validates/rebuilds preprocessing before using current GUI settings."""
import numpy as np
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
