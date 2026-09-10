"""Patch-size edits must reach new runs without mixing cached accumulators."""
from dataclasses import replace
import json

import numpy as np
import pytest

from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.io.ser import SERSource
from test_local_alignment import capture, config, scene


@pytest.mark.parametrize('size', [True, 14, 32, 256, 65., '33', None])
def test_invalid_size_is_refused(size):
    with pytest.raises(ValueError, match='odd integer'):
        ReconstructionConfig(local_patch_size=size)


def test_older_saved_settings_keep_65_pixel_default():
    saved = config().to_dict()
    saved.pop('local_patch_size')
    assert ReconstructionConfig.from_dict(saved).local_patch_size == 65
    selected = config(local_patch_size=33)
    assert ReconstructionConfig.from_dict(selected.to_dict()) == selected


def test_size_changes_output_reuses_cache_and_resumes_exactly(tmp_path):
    with SERSource(capture(tmp_path, 8)) as source:
        ordinary = config()
        small = replace(ordinary, local_patch_size=33)
        preprocess_source(source, ordinary)
        cache = source.path.with_name(source.path.name+'.planetrecon-preprocess.npz')
        cache_bytes = cache.read_bytes()
        previous = stack_source(source, ordinary, state_checkpoint=tmp_path/'old.npz')
        whole = stack_source(source, small)
        assert whole.n_used == previous.n_used
        assert whole.provenance['reference_index'] == previous.provenance['reference_index']
        assert whole.provenance['local_alignment']['window_px'] == 33
        assert whole.provenance['local_alignment']['step_px'] == 16
        assert not np.array_equal(whole.image, previous.image)
        assert whole.channel_order == 'RGB' and whole.validity[30:65, 35:75].all()
        with np.load(tmp_path/'old.npz') as data:
            identity = json.loads(str(data['metadata']))['identity']
        assert 'local_patch_size' not in identity['config']
        assert 'local_patch_size' not in identity['capture']['config']
        with pytest.raises(ValueError, match='identity'):
            stack_source(source, small, resume_from=tmp_path/'old.npz')
        stop = False
        def event(result, info):
            nonlocal stop
            stop = info['n_processed'] >= 4
        checkpoint = tmp_path/'small.npz'
        partial = stack_source(source, small, should_cancel=lambda: stop,
                               on_event=event, state_checkpoint=checkpoint)
        assert partial.incomplete
        resumed = stack_source(source, small, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        assert cache.read_bytes() == cache_bytes


@pytest.mark.parametrize('color_id', [0, 8])
def test_oversized_patch_is_refused_before_processing(tmp_path, monkeypatch, color_id):
    with SERSource(capture(tmp_path, color_id)) as source:
        cfg = config(local_patch_size=129)
        preprocess_source(source, cfg)
        checkpoint = tmp_path/'existing.npz'
        checkpoint.write_bytes(b'keep existing result')
        events = []
        with monkeypatch.context() as patch:
            def read(index):
                pytest.fail('unavailable local alignment must fail before reading processing frames')
            patch.setattr(source, 'read_raw', read)
            with pytest.raises(ValueError, match='size 129 needs frames at least 135 by 135.*15 to 89'):
                stack_source(source, cfg, state_checkpoint=checkpoint,
                             on_event=lambda *args: events.append(args))
        assert not events and checkpoint.read_bytes() == b'keep existing result'
        global_result = stack_source(source, replace(cfg, local_alignment=False))
        assert global_result.n_used > 0 and not global_result.incomplete


@pytest.mark.parametrize('shape,remedy', [((20, 24), 'Disable Local'), ((38, 45), '15 to 31')])
def test_small_capture_gives_fitting_size_or_global_guidance(tmp_path, shape, remedy):
    from planetrecon.io.ser import write_ser
    path = write_ser(tmp_path/'small.ser', np.ones((4, *shape), dtype='u2'))
    with SERSource(path) as source:
        with pytest.raises(ValueError, match=remedy):
            stack_source(source, config())


def test_too_few_selected_frames_are_refused(tmp_path, monkeypatch):
    with SERSource(capture(tmp_path)) as source:
        cfg = config(stack_percent=1, frame_selection_mode='frame_count')
        preprocess_source(source, cfg)
        with monkeypatch.context() as patch:
            def backend(*args, **kwargs):
                pytest.fail('not enough selected frames must fail before starting the stacking backend')
            patch.setattr('planetrecon.pipeline.baseline.select_backend', backend)
            with pytest.raises(ValueError, match='at least four selected frames'):
                stack_source(source, cfg)
        result = stack_source(source, replace(cfg, local_alignment=False))
        assert result.n_used == 1


def test_exact_minimum_size_and_four_frames_still_run(tmp_path):
    from planetrecon.io.ser import write_ser
    with SERSource(capture(tmp_path)) as original:
        frames = np.stack([original.read_raw(i)[:95, :111] for i in range(4)])
    path = write_ser(tmp_path/'minimum.ser', frames.astype('u2'))
    with SERSource(path) as source:
        cfg = config(local_patch_size=89)  # 89 + 6 = 95, exactly the detector height.
        selection = preprocess_source(source, cfg)
        assert selection.accepted.sum() == 4
        result = stack_source(source, cfg)
        assert result.n_used == 4 and result.provenance['local_alignment']['enabled']


def test_gui_commits_typed_size_and_preserves_disabled_choice():
    from PySide6.QtWidgets import QApplication
    from planetrecon.gui.controls import ConfigControls
    app = QApplication.instance() or QApplication([])
    controls = ConfigControls(config())
    try:
        size = controls.fields['local_patch_size']
        assert size.value() == 65 and size.isEnabled() and size.toolTip()
        size.lineEdit().setText('49')
        assert controls.configuration().local_patch_size == 49
        controls.setEnabled(False)
        assert controls.configuration().local_patch_size == 49
        controls.setEnabled(True)
        controls.fields['local_alignment'].setChecked(False)
        assert not size.isEnabled()
        controls.fields['local_alignment'].setChecked(True)
        assert size.isEnabled() and size.value() == 49
        size.lineEdit().setText('32')
        with pytest.raises(ValueError, match='odd integer'):
            controls.configuration()
    finally:
        controls.close()


def test_cli_size_reaches_stack(tmp_path):
    from planetrecon.cli import main
    from planetrecon.result import load_snapshot
    path = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, config())
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--local-patch-size', '33', '--stack-percent', '100',
                 '--out', str(tmp_path/'result')]) == 0
    assert load_snapshot(tmp_path/'result/stack.npz').provenance['local_alignment']['window_px'] == 33


@pytest.mark.hardware
@pytest.mark.parametrize('size', [33, 97])
def test_selected_sizes_match_on_cuda(size):
    import torch
    from scipy.ndimage import map_coordinates
    from planetrecon.pipeline.local_align import LocalRegistration
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    reference = scene()
    y,x = np.indices(reference.shape)
    frame = map_coordinates(reference, [y+.25, x-.8*np.sin(y/25)], order=3, mode='reflect')
    args = dict(window=size, step=size//2)
    cpu = LocalRegistration(reference, **args).displacement(frame, (0., 0.))
    gpu = LocalRegistration(reference, use_cuda=True, **args).displacement(frame, (0., 0.))
    assert np.max(np.abs(cpu)) > .05
    np.testing.assert_allclose(cpu, gpu, atol=1e-10, rtol=1e-10)
