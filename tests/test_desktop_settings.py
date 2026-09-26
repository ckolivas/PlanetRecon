"""Desktop preferences retain controls and distinguish measured/manual geometry."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from types import SimpleNamespace

import pytest

from planetrecon.gui.app import MainWindow, create_app
from planetrecon.gui import settings
from planetrecon.reconstruction import ReconstructionConfig


def test_settings_roundtrip_and_new_capture(tmp_path):
    app = create_app([])
    path = tmp_path / 'preferences.json'
    capture = tmp_path / 'Sat-L.ser'
    capture.touch()
    win = MainWindow(capture, ReconstructionConfig(), settings_path=path)
    fields = win.controls.fields
    fields['device'].setCurrentText('gpu')
    fields['geometry_mode'].setCurrentText('saturn')
    fields['ring_inner_radius_px'].setText('120.25')
    fields['pole_pa_rad'].setText('180.34')
    win.controls.geometry_manual.add('pole_pa_rad')
    fields['equatorial_radius_px'].setText('93.3')
    win.controls.geometry_auto['equatorial_radius_px'] = '93.3'
    fields['geometry_mode'].setCurrentText('none')
    win.save_gamma.setText('1.2')
    win.encoding.setCurrentText('png16')
    assert not win.quality_plot.absolute_control.isChecked()
    assert not win.quality_plot.log_control.isChecked()
    win.quality_plot.absolute_control.setChecked(True)
    win.quality_plot.log_control.setChecked(True)
    win._shutdown()
    restored = MainWindow(capture, settings_path=path)
    assert restored.path == capture
    assert restored.controls.fields['device'].currentData() == 'gpu'
    assert restored.controls.fields['ring_inner_radius_px'].text() == '120.25'
    assert restored.controls.fields['pole_pa_rad'].text() == '180.34'
    assert restored.controls.geometry_manual == win.controls.geometry_manual
    assert restored.controls.geometry_auto['equatorial_radius_px'] == '93.3'
    assert restored.encoding.currentText() == 'png16'
    assert restored.save_gamma.text() == '1.2'
    assert restored.quality_plot.absolute_control.isChecked()
    assert restored.quality_plot.log_control.isChecked()
    other = MainWindow(tmp_path / 'Jup.ser', settings_path=path)
    assert other.controls.fields['equatorial_radius_px'].text() == ''
    assert other.controls.fields['pole_pa_rad'].text() == '180.34'
    assert not other.resume_check.isChecked()
    assert not other.checkpoint_path.text()
    for w in (win, restored, other):
        w.settings_path = None
        w.window.close()
    app.processEvents()


def test_corrupt_settings_and_atomic_write(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text('{broken')
    assert settings.load(path)[1]
    settings.save(path, {'controls': {'fields': {'device': 'cpu'}}})
    assert settings.load(path)[0]['controls']['fields']['device'] == 'cpu'
    assert list(tmp_path.iterdir()) == [path]


def test_control_changes_are_saved_without_closing(tmp_path):
    import time
    app = create_app([])
    path = tmp_path / 'preferences.json'
    win = MainWindow(config=ReconstructionConfig(), settings_path=path)
    win.controls.fields['stack_percent'].setValue(73)
    win.quality_plot.log_control.setChecked(True)
    deadline = time.monotonic() + 3
    while not path.exists() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert settings.load(path)[0]['controls']['fields']['stack_percent'] == 73
    assert settings.load(path)[0]['quality_plot'] == {'absolute': False, 'log': True}
    win.window.close()
    app.processEvents()


def test_file_dialog_directory_survives_restart(monkeypatch, tmp_path):
    app = create_app([])
    preferences = tmp_path / 'preferences.json'
    captures = tmp_path / 'captures'
    captures.mkdir()
    capture = captures / 'Jupiter.ser'
    capture.touch()
    exports = tmp_path / 'exports'
    exports.mkdir()
    win = MainWindow(config=ReconstructionConfig(), settings_path=preferences)
    monkeypatch.setattr('planetrecon.gui.app.QFileDialog.getOpenFileName',
                        lambda *args: (str(capture), ''))
    win._choose()
    assert win._dialog_directory() == str(captures)
    monkeypatch.setattr('planetrecon.gui.app.QFileDialog.getSaveFileName',
                        lambda *args: (str(exports / 'checkpoint.npz'), ''))
    win._choose_checkpoint()
    win.window.close()
    restored = MainWindow(settings_path=preferences)
    try:
        assert restored.last_directory == str(exports)
        seen = []
        def cancel(*args):
            seen.append(args[2])
            return '', ''
        monkeypatch.setattr('planetrecon.gui.app.QFileDialog.getOpenFileName', cancel)
        restored._choose()
        assert seen == [str(exports)]
        assert restored.last_directory == str(exports)
        exports.rmdir()
        assert restored.path is None
        assert restored._dialog_directory() == str(Path.cwd())
    finally:
        restored.window.close()
        app.processEvents()


@pytest.mark.parametrize('encoding,extension', [
    ('tiff32', '.tif'), ('tiff32_raw', '.tif'), ('tiff16', '.tif'), ('png16', '.png')])
def test_save_suggests_result_capture_name(monkeypatch, tmp_path, encoding, extension):
    app = create_app([])
    win = MainWindow(tmp_path / 'new-capture.ser', ReconstructionConfig())
    win.last_result = SimpleNamespace(
        provenance={'source': {'path': str(tmp_path / 'Jupiter.capture.ser')}}, incomplete=False)
    win.encoding.setCurrentText(encoding)
    seen = []
    saved = []
    def choose(*args, **kwargs):
        seen.append(args[2])
        return str(tmp_path / 'chosen-output'), ''
    monkeypatch.setattr('planetrecon.gui.app.QFileDialog.getSaveFileName', choose)
    monkeypatch.setattr(win, 'save_result', lambda path, cfg, **kwargs: saved.append(path))
    try:
        win._choose_save()
        assert seen == [str(tmp_path / ('Jupiter.capture' + extension))]
        assert saved == [tmp_path / ('chosen-output' + extension)]
    finally:
        win.window.close()
        app.processEvents()
