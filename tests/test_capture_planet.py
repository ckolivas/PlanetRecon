"""Capture selection connects published rotation presets to the GUI workflow."""
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from planetrecon.geometry.rotation import planet_from_capture_name, rotation_preset
from planetrecon.gui.app import MainWindow, create_app
from planetrecon.gui.controls import ConfigControls
from planetrecon.reconstruction import ReconstructionConfig

JUPITER = '2022-09-10-0649_2-GB-L-Jup_ZWO ASI224MC.ser'


@pytest.mark.parametrize('name,planet', [
    (JUPITER, 'jupiter'), ('2023-10-10-1320_8-CK-L-Sat_pipp.ser', 'saturn'),
    ('2025-01-10-1459_5-CK-L3-Mars.ser', 'mars'), ('JUPITER-01.SER', 'jupiter'),
    ('neptune.ser', 'neptune'), ('Jupiter-and-Saturn.ser', None),
    ('saturated.ser', None), ('2023-Mar-01.ser', None),
    ('/captures/Jupiter/unknown.ser', None), ('capture.ser', None),
])
def test_only_unambiguous_basename_tokens_select_a_planet(name, planet):
    assert planet_from_capture_name(name) == planet


def test_automatic_planet_follows_capture_but_manual_selection_including_none_wins():
    app = create_app(['capture-planet'])
    controls = ConfigControls(ReconstructionConfig())
    try:
        chooser = controls.fields['rotation_planet']
        controls.suggest_capture_planet(JUPITER)
        assert controls.configuration().rotation_planet == 'jupiter'
        assert 'filename' in controls.planet_identification_label.text()
        controls.suggest_capture_planet('saturn.ser')
        assert controls.configuration().rotation_planet == 'saturn'
        controls.suggest_capture_planet('unknown.ser')
        assert controls.configuration().rotation_planet is None
        controls.suggest_capture_planet(JUPITER)
        chooser.setCurrentIndex(chooser.findData(None))
        controls.suggest_capture_planet(JUPITER)
        controls.suggest_capture_planet('Saturn.ser')
        assert controls.configuration().rotation_planet is None
        chooser.setCurrentIndex(chooser.findData('mars'))
        controls.suggest_capture_planet('Jupiter.ser')
        assert controls.configuration().rotation_planet == 'mars'
    finally:
        controls.close()
        app.processEvents()


def test_unresolved_jupiter_preprocessing_can_start_surface_job(monkeypatch):
    import planetrecon.gui.app as gui
    app = create_app(['jupiter-motion'])
    win = MainWindow(path=Path(JUPITER))
    calls = []
    def start(path, cfg, **options):
        calls.append(cfg)
        raise ValueError('worker reached')
    monkeypatch.setattr(gui, 'start_stack_job', start)
    try:
        win.controls.fields['geometry_mode'].setCurrentText('surface')
        estimate = {'status': 'unresolved', 'suggestions': {},
                    'notes': ['Surface rotation is unresolved; no surface-rate prefill.']}
        win.preprocessing_info = {'status': 'ready', 'geometry_estimate': estimate}
        win.controls.prefill_geometry(estimate)
        win._run()
        assert len(calls) == 1
        cfg = calls[0]
        assert cfg.rotation_planet == 'jupiter' and cfg.surface_rate_rad_s is None
        assert cfg.effective_surface_rate() == rotation_preset('jupiter')['surface_rate_rad_s']
        assert cfg.sub_obs_lat_rad is None
        assert win.error.text() == 'worker reached'
        assert win.error.textInteractionFlags() & Qt.TextSelectableByMouse
        assert win.error.textInteractionFlags() & Qt.TextSelectableByKeyboard
        assert win.error.textFormat() == Qt.PlainText
        win.controls.fields['surface_rate_rad_s'].setText('0')
        win._run()
        assert calls[-1].effective_surface_rate() == 0.
    finally:
        win._shutdown()
        win.window.close()
        app.processEvents()


def test_start_respects_explicit_preset_and_resume_without_filename_inference(monkeypatch):
    import planetrecon.gui.app as gui
    app = create_app(['manual-planet'])
    win = MainWindow(path=Path(JUPITER), config=ReconstructionConfig(rotation_planet='mars'))
    try:
        assert win.controls.configuration().rotation_planet == 'mars'
    finally:
        win._shutdown()
        win.window.close()
    win = MainWindow(config=ReconstructionConfig(geometry_mode='surface', surface_rate_rad_s=.01))
    win.path = Path(JUPITER)
    win.resume_check.setChecked(True)
    win.checkpoint_path.setText('checkpoint.npz')
    calls = []
    def start(path, cfg, **options):
        calls.append(cfg)
        raise ValueError('worker reached')
    monkeypatch.setattr(gui, 'start_stack_job', start)
    try:
        win._run()
        assert calls[0].rotation_planet is None
    finally:
        win._shutdown()
        win.window.close()
        app.processEvents()
