from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.io.ser import SERSource, write_ser
from planetrecon.pipeline import baseline, local_align
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.reconstruction import ReconstructionConfig


def capture(path, colour):
    yy, xx = np.indices((96, 112))
    disk = ((xx-56)/35)**2 + ((yy-48)/32)**2 < 1
    scene = disk*(3000+300*np.cos(xx/2)+400*np.sin(yy/3))
    frames = np.stack([gaussian_filter(scene, sigma) for sigma in np.linspace(.6, 1.4, 12)])
    if colour == 8:
        frames[:, ::2, ::2] *= .8
        frames[:, 1::2, 1::2] *= .6
    elif colour == 100:
        frames = frames[..., None]*[.8, 1., .6]
    return write_ser(path, frames.astype('u2'), color_id=colour)


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=2,
                                local_alignment=True, **kwargs)


@pytest.mark.parametrize('colour', [0, 8, 100])
def test_squared_weights_match_independent_sums_with_identical_registration(tmp_path, monkeypatch, colour):
    path = capture(tmp_path/'weights.ser', colour)
    with SERSource(path) as source:
        cfg = config()
        selection = preprocess_source(source, cfg)
        cache = path.with_name(path.name+'.planetrecon-preprocess.npz')
        cache_before = cache.read_bytes()
        projections, shifts = [], []
        original = local_align.cpu_backproject
        def record(raw, shift, color):
            projected = original(raw, shift, color)
            projections.append((projected[0].copy(), projected[1].copy()))
            shifts.append(np.array(shift))
            return projected
        monkeypatch.setattr(local_align, 'cpu_backproject', record)
        linear = baseline.stack_source(source, cfg)
        count = len(projections)
        q = selection.measurements[selection.accepted, 0]
        assert count == linear.n_used == len(q)
        numerator = np.zeros_like(linear.image)
        denominator = np.zeros_like(linear.coverage)
        for quality, (add, support) in zip(q, projections):
            numerator += quality*quality*add
            denominator += quality*quality*support
        expected = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        squared = baseline.stack_source(source, replace(cfg, squared_quality_weights=True))
        native = denominator > 0
        np.testing.assert_allclose(squared.image[native], expected[native], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(squared.coverage[native], denominator[native])
        np.testing.assert_array_equal(shifts[:count], shifts[count:])
        assert squared.n_used == linear.n_used and squared.n_rejected == linear.n_rejected
        assert squared.provenance['reference_index'] == linear.provenance['reference_index']
        assert squared.provenance['local_alignment'] == linear.provenance['local_alignment']
        assert squared.provenance['scalar_frame_weight'] == 'squared cached quality'
        assert not np.array_equal(squared.image, linear.image)
        assert cache.read_bytes() == cache_before


@pytest.mark.parametrize('stronger', [False, True])
def test_weight_checkpoint_resumes_exactly_and_rejects_changed_setting(tmp_path, stronger):
    path = capture(tmp_path/'resume.ser', 8)
    cfg = config(squared_quality_weights=stronger)
    with SERSource(path) as source:
        preprocess_source(source, cfg)
        whole = baseline.stack_source(source, cfg)
        stop = False
        def event(result, info):
            nonlocal stop
            stop = info['n_processed'] >= 4
        checkpoint = tmp_path/'state.npz'
        partial = baseline.stack_source(source, cfg, on_event=event,
                                        should_cancel=lambda: stop, state_checkpoint=checkpoint)
        assert partial.incomplete
        import json
        with np.load(checkpoint) as data:
            identity = json.loads(str(data['metadata']))['identity']
        for settings in (identity['config'], identity['capture']['config']):
            # Unchanged linear sums keep the pre-option checkpoint identity.
            assert ('squared_quality_weights' in settings) is stronger
        resumed = baseline.stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        with pytest.raises(ValueError, match='identity|configuration'):
            baseline.stack_source(source, replace(cfg, squared_quality_weights=not stronger), resume_from=checkpoint)


def test_squared_weights_are_optional_and_require_supported_configuration():
    assert not ReconstructionConfig().squared_quality_weights
    with pytest.raises(ValueError, match='squared_quality_weights must be a bool'):
        config(squared_quality_weights=1)
    with pytest.raises(ValueError, match='require experimental local alignment'):
        ReconstructionConfig(squared_quality_weights=True)
    cfg = config(squared_quality_weights=True)
    assert ReconstructionConfig.from_dict(cfg.to_dict()) == cfg


def test_gui_preserves_stronger_weights_while_busy_and_gates_incompatible_modes():
    from PySide6.QtWidgets import QApplication
    from planetrecon.gui.controls import ConfigControls
    app = QApplication.instance() or QApplication([])
    controls = ConfigControls(ReconstructionConfig())
    try:
        stronger = controls.fields['squared_quality_weights']
        local = controls.fields['local_alignment']
        assert not stronger.isChecked() and not stronger.isEnabled()
        local.setChecked(True)
        stronger.setChecked(True)
        assert controls.configuration().squared_quality_weights
        controls.setEnabled(False)
        assert controls.configuration().squared_quality_weights
        controls.setEnabled(True)
        local.setChecked(False)
        assert not controls.configuration().squared_quality_weights
        local.setChecked(True)
        assert controls.configuration().squared_quality_weights
        controls.fields['frame_preselection'].setChecked(False)
        assert not controls.configuration().squared_quality_weights
        controls.fields['frame_preselection'].setChecked(True)
        geometry = controls.fields['geometry_mode']
        geometry.setCurrentIndex(geometry.findData('field'))
        assert not controls.configuration().squared_quality_weights
        geometry.setCurrentIndex(geometry.findData('none'))
        assert controls.configuration().squared_quality_weights
        stronger.setChecked(False)
        assert not controls.configuration().squared_quality_weights
    finally:
        controls.close()


def test_cli_passes_stronger_weighting_to_fresh_stack(tmp_path):
    from planetrecon.cli import main
    from planetrecon.result import load_snapshot
    path = capture(tmp_path/'cli.ser', 8)
    with SERSource(path) as source:
        preprocess_source(source, config())
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--local-alignment', '--squared-quality-weights', '--out', str(tmp_path/'result')]) == 0
    result = load_snapshot(tmp_path/'result/stack.npz')
    assert result.provenance['scalar_frame_weight'] == 'squared cached quality'


@pytest.mark.hardware
def test_squared_quality_cpu_cuda_agree(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    with SERSource(capture(tmp_path/'cuda.ser', 8)) as source:
        cfg = config(squared_quality_weights=True)
        selection = preprocess_source(source, cfg)
        cpu = baseline.stack_source(source, cfg)
        gpu = baseline.stack_source(source, replace(cfg, device='gpu'))
        assert gpu.backend == 'cuda' and gpu.n_used == cpu.n_used
        np.testing.assert_allclose(gpu.image, cpu.image, rtol=1e-10, atol=1e-7)
        # Coverage carries arbitrary quality-score units. Compare fractions of
        # available weight so near-zero CFA support has a meaningful tolerance.
        total_weight = np.sum(selection.measurements[selection.accepted, 0]**2)
        np.testing.assert_allclose(gpu.coverage/total_weight, cpu.coverage/total_weight,
                                   rtol=1e-10, atol=1e-12)
