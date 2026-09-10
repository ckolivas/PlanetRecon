"""Rejected interpolation jobs must not silently become ordinary stacks."""
import pytest

from planetrecon.reconstruction import ReconstructionConfig


def test_saved_ordinary_job_retains_its_settings():
    expected = ReconstructionConfig(stack_percent=63, frame_selection_mode='quality_range')
    saved = {**expected.to_dict(), 'local_cfa_interpolation': False}
    assert ReconstructionConfig.from_dict(saved) == expected
    assert saved['local_cfa_interpolation'] is False
    assert 'local_cfa_interpolation' not in expected.to_dict()


def test_saved_interpolation_job_is_refused():
    saved = {**ReconstructionConfig().to_dict(), 'local_cfa_interpolation': True}
    with pytest.raises(ValueError, match='removed after artifact review'):
        ReconstructionConfig.from_dict(saved)


def test_cli_does_not_offer_rejected_interpolation(capsys):
    from planetrecon.cli import main
    with pytest.raises(SystemExit) as exc:
        main(['stack', '--help'])
    assert exc.value.code == 0
    assert '--local-cfa-interpolation' not in capsys.readouterr().out


def test_gui_keeps_ordinary_local_defaults():
    from PySide6.QtWidgets import QApplication
    from planetrecon.gui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    controls = window.controls
    try:
        assert 'local_cfa_interpolation' not in controls.fields
        config = controls.configuration()
        assert config.local_alignment and config.stack_percent == 50
        assert config.frame_selection_mode == 'quality_range'
        assert 'squared_quality_weights' not in controls.fields
        assert 'squared_quality_weights' not in config.to_dict()
    finally:
        window.window.close()
