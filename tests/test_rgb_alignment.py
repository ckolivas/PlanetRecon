from dataclasses import replace
import hashlib

import numpy as np
import pytest
import tifffile

from planetrecon.postprocess import align_rgb
from planetrecon.result import ReconstructionResult, load_snapshot, save_snapshot


def original():
    image = np.zeros((12, 16, 3))
    image[5, 7] = [3., 5., 7.]
    return ReconstructionResult(image, np.ones_like(image), np.ones_like(image, dtype=bool),
        'adu', 'RGB', 'cpu', 'float64', 'final', False, n_used=10,
        layer_coverage={'globe': np.ones((12, 16)), 'cfa_direct_R': np.ones((12, 16))*2})


def test_channel_motion_sign_isolation_and_no_wrap():
    source = original()
    moved = align_rgb(source, red=(2, -1), blue=(-3, 2))
    assert moved.image[4, 9, 0] == 3 and moved.image[7, 4, 2] == 7
    np.testing.assert_array_equal(moved.image[..., 1], source.image[..., 1])
    assert not moved.validity[:, :2, 0].any()
    assert not moved.validity[-1, :, 0].any()
    assert not moved.validity[:, -3:, 2].any()
    assert moved.image.sum() == source.image.sum()
    assert source.image[5, 7, 0] == 3 and 'rgb_alignment' not in source.provenance
    assert moved.layer_coverage['globe'].shape == moved.image.shape[:2]
    np.testing.assert_array_equal(moved.layer_coverage['rgb_alignment_R_globe'], moved.validity[..., 0])
    assert moved.layer_coverage['cfa_direct_R'].shape == moved.image.shape[:2]
    np.testing.assert_array_equal(moved.layer_coverage['cfa_direct_R'], 2*moved.validity[..., 0])


def test_fractional_shift_requires_every_input_sample_and_keeps_constants():
    source = original()
    source.image[:] = [3, 5, 7]
    source.validity[5, 7, 0] = False
    source.image[5, 7, 0] = np.nan
    moved = align_rgb(source, red=(.5, 0))
    assert not moved.validity[5, 7:9, 0].any()
    assert not moved.validity[:, 0, 0].any()
    assert np.isfinite(moved.image).all()
    np.testing.assert_array_equal(moved.image[..., 0][moved.validity[..., 0]], 3.)
    np.testing.assert_array_equal(moved.image[..., 1], source.image[..., 1])
    assert not np.shares_memory(moved.image, source.image)


def test_zero_offsets_are_exact_and_repeated_resampling_is_refused():
    source = original()
    reset = align_rgb(source)
    for key in ('image', 'coverage', 'validity'):
        np.testing.assert_array_equal(getattr(reset, key), getattr(source, key))
    assert reset.provenance == source.provenance
    with pytest.raises(ValueError, match='original result'):
        align_rgb(align_rgb(source, red=(.5, 0)), red=(.5, 0))


@pytest.mark.parametrize('offset', [(float('nan'), 0), (0, float('inf')), (16.1, 0)])
def test_invalid_offsets_are_refused(offset):
    with pytest.raises(ValueError, match='finite'):
        align_rgb(original(), red=offset)


@pytest.mark.parametrize('changes', [{'incomplete': True}, {'spatial_stride': 2}, {'channel_order': 'mono'}])
def test_only_complete_full_rgb_results_can_be_adjusted(changes):
    with pytest.raises(ValueError, match='complete full-resolution RGB'):
        align_rgb(replace(original(), **changes))


def test_cli_exports_adjusted_image_and_preserves_input(tmp_path):
    from planetrecon.cli import main
    source = tmp_path/'input.npz'
    save_snapshot(source, original())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    args = ['align-rgb', '--input', str(source), '--out', str(tmp_path/'adjusted'),
            '--red', '2', '-1', '--blue', '-3', '2']
    assert main(args) == 0
    saved = load_snapshot(tmp_path/'adjusted/aligned.npz')
    exported = tifffile.imread(tmp_path/'adjusted/aligned.tif')
    np.testing.assert_array_equal(exported[saved.validity], saved.image.astype(np.float32)[saved.validity])
    assert np.isnan(exported[~saved.validity]).all()
    assert saved.provenance['rgb_alignment']['offsets_xy_px']['R'] == [2., -1.]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_gui_repreviews_from_original_cancel_restores_and_new_run_resets(monkeypatch, tmp_path):
    from planetrecon.gui.app import MainWindow, create_app
    from planetrecon.jobs import result_payload
    import planetrecon.gui.rgb_alignment as dialog
    app = create_app(['rgb-test'])
    window = MainWindow()
    try:
        saved = tmp_path/'original.npz'
        save_snapshot(saved, original())
        config = window.controls.configuration()
        window._load_result(saved)
        assert window.controls.configuration() == config
        window.show()
        for _ in range(8):
            app.processEvents()
        assert window.image_label.pixmap().height() <= window.canvas.viewport().height()
        assert window.align_rgb_btn.isEnabled()
        base = window.unaligned_result
        window._apply_rgb_offsets(((.5, 0), (0, 0)))
        window._apply_rgb_offsets(((1, 0), (0, 0)))
        np.testing.assert_array_equal(window.last_result.image, align_rgb(base, red=(1, 0)).image)
        previous = window.last_result
        def cancel(parent, offsets, apply):
            apply(((2, 0), (0, 0)))
            return False
        monkeypatch.setattr(dialog, 'edit_rgb_alignment', cancel)
        window._align_rgb()
        assert window.last_result is previous and window.rgb_offsets == ((1, 0), (0, 0))
        window._apply_rgb_offsets(((0, 0), (0, 0)))
        np.testing.assert_array_equal(window.last_result.image, base.image)
        assert 'rgb_alignment' not in window.last_result.provenance
        window._apply_rgb_offsets(((1, 0), (0, 0)))
        window._accept_result(result_payload(original()))
        assert window.rgb_offsets == ((0., 0.), (0., 0.))
        assert 'rgb_alignment' not in window.last_result.provenance
        window._accept_result(result_payload(replace(original(), incomplete=True)))
        assert not window.align_rgb_btn.isEnabled()
        adjusted = tmp_path/'already-aligned.npz'
        save_snapshot(adjusted, align_rgb(original(), red=(1, 0)))
        window._load_result(adjusted)
        assert not window.align_rgb_btn.isEnabled()
        assert 'original NPZ' in window.result_label.text()
    finally:
        window.window.close()
        app.processEvents()


def test_dialog_apply_reset_and_cancel_commit_typed_values():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QDoubleSpinBox
    from planetrecon.gui.app import create_app
    from planetrecon.gui.rgb_alignment import edit_rgb_alignment
    app = create_app(['rgb-dialog-test'])
    calls = []
    failures = []
    def interact():
        dialog = QApplication.activeModalWidget()
        try:
            red_x = dialog.findChild(QDoubleSpinBox, 'red_x')
            red_x.lineEdit().setText('0.75')
            buttons = dialog.findChild(QDialogButtonBox)
            buttons.button(QDialogButtonBox.StandardButton.Apply).click()
            assert calls[-1] == ((.75, 0.), (0., 0.))
            buttons.button(QDialogButtonBox.StandardButton.Reset).click()
            assert calls[-1] == ((0., 0.), (0., 0.))
            assert all(button.toolTip() for button in buttons.buttons())
        except Exception as exc:
            failures.append(exc)
        finally:
            dialog.reject()
    QTimer.singleShot(0, interact)
    assert not edit_rgb_alignment(None, ((0., 0.), (0., 0.)), calls.append)
    assert not failures
