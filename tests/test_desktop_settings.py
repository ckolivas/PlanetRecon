"""Desktop preferences retain controls and distinguish measured/manual geometry."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path

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
    win._shutdown()
    restored = MainWindow(settings_path=path)
    assert restored.path == capture
    assert restored.controls.fields['device'].currentData() == 'gpu'
    assert restored.controls.fields['ring_inner_radius_px'].text() == '120.25'
    assert restored.controls.fields['pole_pa_rad'].text() == '180.34'
    assert restored.controls.geometry_manual == win.controls.geometry_manual
    assert restored.controls.geometry_auto['equatorial_radius_px'] == '93.3'
    assert restored.encoding.currentText() == 'png16'
    assert restored.save_gamma.text() == '1.2'
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
