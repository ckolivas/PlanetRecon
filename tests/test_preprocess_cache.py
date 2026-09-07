"""Independent caching, optional reuse, invalidation and exclusion UI."""
from dataclasses import replace
import json
import os
import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import (
    preprocess_source, load_cache, default_cache_path, exclusion_counts,
)
from planetrecon.reconstruction import ReconstructionConfig


def capture(tmp_path):
    y, x = np.indices((80, 96))
    normal = gaussian_filter((((x-48)/22)**2 + ((y-40)/18)**2 < 1)*(100+15*np.cos(y)), .7)
    wide = gaussian_filter((((x-48)/32)**2 + ((y-40)/18)**2 < 1)*(100+15*np.cos(y)), .7)
    frames = np.stack([normal]*32 + [gaussian_filter(normal, 6), wide])
    return write_ser(tmp_path/'capture.ser', frames.astype('u2'))


def config(**kw):
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=8, **kw)


def test_cache_is_optional_and_runs_never_implicitly_preprocess(tmp_path, monkeypatch):
    path = capture(tmp_path)
    cfg = config()
    with SERSource(path) as source:
        assert not default_cache_path(source).exists()
        unfiltered = stack_source(source, cfg)
        assert unfiltered.n_used == 34
        selected = preprocess_source(source, cfg)
        assert default_cache_path(source).is_file()
        assert exclusion_counts(selected)['excluded'] == 2
    import planetrecon.pipeline.preprocess_cache as module
    monkeypatch.setattr(module, 'screen_source', lambda *a, **k: pytest.fail('run repeated preprocessing'))
    with SERSource(path) as source:
        loaded, report = load_cache(source, cfg)
        assert report['status'] == 'ready'
        np.testing.assert_array_equal(selected.measurements, loaded.measurements)
        filtered = stack_source(source, replace(cfg, batch_frames=3, max_shift_px=50))
        disabled = stack_source(source, replace(cfg, frame_preselection=False))
        assert filtered.n_used == 32 and filtered.n_rejected == 2
        assert filtered.provenance['preprocessing_cache']['digest'] == selected.digest
        np.testing.assert_array_equal(unfiltered.image, disabled.image)
        assert disabled.n_used == 34


def test_full_pixel_identity_detects_changed_middle_with_preserved_file_times(tmp_path):
    path = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, config())
    stat = path.stat()
    with path.open('r+b') as stream:
        stream.seek(stat.st_size//2)
        value = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([value[0] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with SERSource(path) as source:
        assert load_cache(source, config())[1]['status'] == 'stale'
        with pytest.raises(ValueError, match='Preprocess again'):
            stack_source(source, config())
        assert stack_source(source, config(frame_preselection=False)).n_used == 34


def test_calibration_and_colour_invalidate_but_device_and_geometry_do_not(tmp_path):
    path = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, config())
        assert load_cache(source, replace(config(), device='gpu', geometry_mode='field'))[1]['status'] == 'ready'
        from planetrecon.calibration import Calibration
        assert load_cache(source, config(), Calibration(gain_e_per_adu=2))[1]['status'] == 'stale'
    with SERSource(path, bayer_override='RGGB') as source:
        assert load_cache(source, config())[1]['status'] == 'stale'


def test_cancel_preserves_cache_and_cache_cannot_replace_input(tmp_path):
    path = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, config())
        cache = default_cache_path(source)
        before = cache.read_bytes()
        with pytest.raises(InterruptedError):
            preprocess_source(source, config(), should_cancel=lambda: True)
        assert cache.read_bytes() == before
        with pytest.raises(ValueError, match='cannot replace'):
            preprocess_source(source, config(), cache_path=path)
        assert load_cache(source, config())[1]['status'] == 'ready'


def test_cli_preprocess_then_optional_cached_stack(tmp_path, capsys):
    from planetrecon.cli import main
    path = capture(tmp_path)
    cache = tmp_path/'chosen-cache.npz'
    assert main(['--threads', '2', 'preprocess', '--path', str(path), '--cache', str(cache)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['quality'] >= 1 and report['shape'] >= 1 and report['excluded'] == 2
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--preprocessing-cache', str(cache), '--out', str(tmp_path/'out')]) == 0
    from planetrecon.result import load_snapshot
    assert load_snapshot(tmp_path/'out/stack.npz').n_used == 32


def test_corrupt_cache_is_reported_and_can_be_rebuilt(tmp_path):
    path = capture(tmp_path)
    with SERSource(path) as source:
        selected = preprocess_source(source, config())
        cache = default_cache_path(source)
        with np.load(cache, allow_pickle=False) as data:
            contents = {key: data[key].copy() for key in data.files}
        contents['accepted'][0] = False
        np.savez_compressed(cache, **contents)
        assert load_cache(source, config())[1]['status'] == 'invalid'
        cache.write_bytes(b'broken cache')
        assert load_cache(source, config())[1]['status'] == 'invalid'
        rebuilt = preprocess_source(source, config())
        assert load_cache(source, config())[1]['status'] == 'ready'
        np.testing.assert_array_equal(selected.accepted, rebuilt.accepted)


def test_geometry_refresh_preserves_accumulator_identity_but_weights_do_not(tmp_path):
    from planetrecon.pipeline.preprocess_cache import reconstruction_digest, selection_digest, cache_report
    path = capture(tmp_path)
    with SERSource(path) as source:
        selected = preprocess_source(source, config())
    original = reconstruction_digest(selected)
    selected.summary['geometry_estimate']['notes'].append('Updated geometry report')
    assert selection_digest(selected) != selected.digest
    assert reconstruction_digest(selected) == original
    report = cache_report(selected, config=replace(config(), cadence_s=1))
    assert report['status'] == 'ready'
    assert report['geometry_estimate']['applicable'] is False
    selected.measurements[0, 0] *= 2
    assert reconstruction_digest(selected) != original


def test_gui_separate_preprocess_shows_counts_and_keeps_existing_result(tmp_path):
    pytest.importorskip('PySide6')
    from planetrecon.gui.app import create_app, MainWindow
    app = create_app(['cache-gui-test'])
    win = MainWindow(path=capture(tmp_path), config=config())
    def wait():
        deadline = time.monotonic()+15
        while win.job is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert win.job is None, win.error.text()
        assert not win.error.text(), win.error.text()
    try:
        assert win.preprocess_btn.toolTip()
        win._preprocess()
        wait()
        assert win.last_result is None
        assert win.preprocessing_info['excluded'] == 2
        assert 'quality /' in win.preprocessing_label.text()
        assert 'shape exclusions' in win.preprocessing_label.text()
        win._run()
        wait()
        assert win.last_result.n_used == 32
        previous = win.last_result
        win.controls.fields['frame_preselection'].setChecked(False)
        assert 'disabled' in win.preprocessing_label.text()
        win._preprocess()
        wait()
        assert win.last_result is previous
        win._run()
        wait()
        assert win.last_result.n_used == 34
    finally:
        win._shutdown()
        win.window.close()
        app.processEvents()
