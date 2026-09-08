from dataclasses import replace
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from planetrecon.pipeline.local_align import LocalRegistration, pull, cpu_backproject
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.io.ser import SERSource, write_ser
from planetrecon.reconstruction import ReconstructionConfig


def scene():
    return gaussian_filter(np.random.default_rng(204).normal(size=(160, 192)), 1.5)*100+300


def test_local_motion_restores_detail_without_sharpening():
    ref = scene()
    yy, xx = np.indices(ref.shape)
    flow = 1.5*np.sin(2*np.pi*yy/160)
    frame = map_coordinates(ref, [yy, xx-flow], order=3, mode='reflect')
    displacement = LocalRegistration(ref, window=49, step=16).displacement(frame, (0., 0.))
    corrected = pull(frame, displacement)
    roi = np.s_[48:-48, 48:-48]
    assert np.linalg.norm((corrected-ref)[roi]) < .25*np.linalg.norm((frame-ref)[roi])
    assert corrected[roi].min() >= frame.min()
    assert corrected[roi].max() <= frame.max()


@pytest.mark.parametrize('gain,offset', [(1., 0.), (.8, 25.), (1.2, -10.)])
def test_brightness_change_does_not_create_motion(gain, offset):
    ref = scene()
    displacement = LocalRegistration(ref).displacement(ref*gain+offset, (0., 0.))
    assert max(np.abs(v).max() for v in displacement) == 0.


def test_blur_change_does_not_create_significant_motion():
    ref = scene()
    displacement = LocalRegistration(ref, window=49, step=16).displacement(gaussian_filter(ref, 1.), (0., 0.))
    roi = np.s_[48:-48, 48:-48]
    assert np.sqrt(np.mean(displacement[0][roi]**2+displacement[1][roi]**2)) < .01


def test_ambiguous_diagonal_texture_and_small_frames_keep_global_motion():
    yy, xx = np.indices((160, 192))
    ref = 300+100*np.sin((xx+yy)/4)
    frame = 300+100*np.sin((xx+yy-1.3)/4)
    displacement = LocalRegistration(ref).displacement(frame, (0., 0.))
    assert max(np.abs(v).max() for v in displacement) == 0.
    small = np.ones((30, 40))
    for d, expected in zip(LocalRegistration(small).displacement(small, (.3, -.2)), (.3, -.2)):
        np.testing.assert_array_equal(d, np.full(small.shape, expected))


def test_unrelated_noise_does_not_drive_warp():
    ref = scene()
    other = gaussian_filter(np.random.default_rng(921).normal(size=ref.shape), 1.5)*100+300
    displacement = LocalRegistration(ref).displacement(other, (0., 0.))
    assert max(np.abs(v).max() for v in displacement) == 0.


@pytest.mark.parametrize('color', ['mono', 'RGB', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
def test_dense_support_preserves_constant_intensity_and_zero_exterior(color):
    from planetrecon.detector import cfa_labels, is_bayer
    yy, xx = np.indices((25, 31))
    displacement = (.3+np.sin(yy/5), -.7+np.cos(xx/5))
    rgb = np.ones((25, 31, 3))*[4., 7., 9.]
    raw = rgb if color == 'RGB' else np.ones((25, 31))*7.
    if is_bayer(color):
        labels = cfa_labels(25, 31, color)
        raw = sum(rgb[..., i]*(labels == c) for i, c in enumerate('RGB'))
    add, weight, demo, support = cpu_backproject(raw, displacement, color)
    np.testing.assert_allclose(add, weight*([4., 7., 9.] if color != 'mono' else 7.), atol=1e-13)
    assert (weight >= 0).all()
    assert (support < 1).any() if support is not None else True


def capture(tmp_path, color_id=0):
    yy, xx = np.indices((96, 112))
    disk = ((xx-56)/35)**2+((yy-48)/32)**2 < 1
    ref = gaussian_filter(disk*(3000+300*np.cos(xx/2)+400*np.sin(yy/3)), .6)
    frames = np.stack([map_coordinates(ref, [yy, xx-.8*np.sin(yy/20)*np.sin(i)],
                                      order=1, mode='constant') for i in range(16)])
    if color_id == 8:
        frames[:, ::2, ::2] *= .8
        frames[:, 1::2, 1::2] *= .6
    if color_id == 100:
        frames = frames[..., None]*[.8, 1., .6]
    return write_ser(tmp_path/'local.ser', frames.astype('u2'), color_id=color_id)


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=2, local_alignment=True, **kwargs)


def test_application_keeps_frames_reuses_cache_and_resumes_exactly(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        cfg = config()
        selected = preprocess_source(source, cfg)
        cache_bytes = source.path.with_name(source.path.name+'.planetrecon-preprocess.npz').read_bytes()
        whole = stack_source(source, cfg)
        global_result = stack_source(source, replace(cfg, local_alignment=False))
        assert whole.n_used == global_result.n_used == selected.accepted.sum()
        assert whole.provenance['reference_index'] == selected.best_reference_index
        assert whole.provenance['local_alignment']['enabled']
        assert not np.array_equal(whole.image, global_result.image)
        assert source.path.with_name(source.path.name+'.planetrecon-preprocess.npz').read_bytes() == cache_bytes
        stop = False
        def event(result, info):
            nonlocal stop
            if info['n_processed'] >= 4:
                stop = True
        checkpoint = tmp_path/'state.npz'
        partial = stack_source(source, cfg, should_cancel=lambda: stop, on_event=event, state_checkpoint=checkpoint)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        with pytest.raises(ValueError, match='identity|configuration'):
            stack_source(source, replace(cfg, local_alignment=False), resume_from=checkpoint)


def test_local_alignment_requires_matching_preprocessing_and_compatible_geometry(tmp_path):
    with pytest.raises(ValueError, match='local alignment'):
        config(frame_preselection=False)
    with pytest.raises(ValueError, match='local alignment'):
        config(geometry_mode='field')
    with SERSource(capture(tmp_path)) as source:
        with pytest.raises(ValueError, match='Preprocess'):
            stack_source(source, config())


def test_gui_local_alignment_is_optional_and_updates_new_run():
    from planetrecon.gui.controls import ConfigControls
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    controls = ConfigControls(ReconstructionConfig())
    try:
        edit = controls.fields['local_alignment']
        assert not edit.isChecked()
        edit.setChecked(True)
        assert controls.configuration().local_alignment
        controls.setEnabled(False)
        assert controls.configuration().local_alignment  # A busy GUI must not change saved values.
        controls.setEnabled(True)
        controls.fields['frame_preselection'].setChecked(False)
        assert not edit.isEnabled() and not controls.configuration().local_alignment
        controls.fields['frame_preselection'].setChecked(True)
        assert controls.configuration().local_alignment
        geo = controls.fields['geometry_mode']
        geo.setCurrentIndex(geo.findData('field'))
        assert not controls.configuration().local_alignment
        geo.setCurrentIndex(geo.findData('none'))
        assert controls.configuration().local_alignment
        edit.setChecked(False)
        assert not controls.configuration().local_alignment
    finally:
        controls.close()


@pytest.mark.hardware
def test_cuda_patch_displacements_match_cpu():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    ref = scene()
    yy, xx = np.indices(ref.shape)
    frame = map_coordinates(ref, [yy+.25, xx-1.5*np.sin(2*np.pi*yy/160)], order=3, mode='reflect')
    cpu = LocalRegistration(ref, window=49, step=16).displacement(frame, (.375, -.625))
    gpu = LocalRegistration(ref, window=49, step=16, use_cuda=True).displacement(frame, (.375, -.625))
    np.testing.assert_allclose(cpu, gpu, atol=1e-10, rtol=1e-10)


@pytest.mark.hardware
@pytest.mark.parametrize('color', ['mono', 'RGB', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
def test_cuda_dense_backprojection_matches_cpu(color):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.backends.torch_accel import TorchBackend
    yy, xx = np.indices((25, 31))
    displacement = (1.2*np.sin(yy/8)+.375, 1.1*np.cos(xx/9)-.625)
    raw = np.random.default_rng(203).uniform(1, 200, (25, 31, 3) if color == 'RGB' else (25, 31))
    for actual, expected in zip(TorchBackend().backproject(raw, displacement, color),
                                cpu_backproject(raw, displacement, color)):
        if actual is not None:
            if actual.ndim == 2 and expected.ndim == 3:
                actual = actual[..., None]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.hardware
@pytest.mark.parametrize('color_id', [0, 8, 100])
def test_cuda_local_stack_matches_cpu_and_keeps_colour(tmp_path, color_id):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    with SERSource(capture(tmp_path, color_id)) as source:
        cfg = config()
        preprocess_source(source, cfg)
        cpu = stack_source(source, cfg)
        gpu = stack_source(source, replace(cfg, device='gpu'))
        assert gpu.backend == 'cuda' and gpu.n_used == cpu.n_used
        np.testing.assert_allclose(cpu.image, gpu.image, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(cpu.coverage, gpu.coverage, rtol=1e-10, atol=1e-8)
        if color_id:
            assert gpu.channel_order == 'RGB' and gpu.validity[30:65, 35:75].all()
            means = gpu.image[30:65, 35:75].mean(axis=(0, 1))
            np.testing.assert_allclose(means/means[1], [.8, 1., .6], atol=.04)


def test_cli_exposes_local_alignment(tmp_path):
    from planetrecon.cli import main
    from planetrecon.result import load_snapshot
    path = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, config())
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--local-alignment', '--out', str(tmp_path/'result')]) == 0
    result = load_snapshot(tmp_path/'result/stack.npz')
    assert result.provenance['local_alignment']['enabled']


def test_cuda_failure_during_matching_keeps_cpu_local_alignment(tmp_path, monkeypatch):
    from planetrecon.backends.cpu import CPUBackend
    from planetrecon.backends.base import DeviceReport
    import planetrecon.pipeline.baseline as baseline
    with SERSource(capture(tmp_path)) as source:
        cfg = config()
        preprocess_source(source, cfg)
        expected = stack_source(source, cfg)
        class FakeCUDA(CPUBackend):
            name = 'cuda'
        backend = FakeCUDA(threads=2)
        monkeypatch.setattr(baseline, 'select_backend', lambda *args, **kwargs:
                            (backend, DeviceReport('gpu', 'cuda', ['cpu', 'gpu'], False)))
        def fail(*args):
            raise RuntimeError('simulated CUDA patch memory failure')
        monkeypatch.setattr(LocalRegistration, '_cuda_costs', fail)
        actual = stack_source(source, replace(cfg, device='gpu'))
        assert actual.backend == 'cpu'
        assert actual.provenance['device_report']['execution_history'] == ['cuda', 'cpu']
        np.testing.assert_array_equal(actual.image, expected.image)


@pytest.mark.hardware
def test_native_rgb_global_cuda_coverage_broadcasts_to_all_channels(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    with SERSource(capture(tmp_path, 100)) as source:
        cfg = replace(config(), local_alignment=False)
        preprocess_source(source, cfg)
        cpu = stack_source(source, cfg)
        gpu = stack_source(source, replace(cfg, device='gpu'))
        assert gpu.backend == 'cuda'
        np.testing.assert_allclose(cpu.image, gpu.image, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(cpu.coverage, gpu.coverage, rtol=1e-10, atol=1e-8)


def test_patch_grid_boundary_cannot_collapse_or_fold_image_coordinates():
    ref = scene()
    matcher = LocalRegistration(ref, use_cuda=True)
    yy, xx = np.indices((7, 7))
    peak = np.exp(-.02*((yy-3)**2+(xx-5)**2))
    # Force a confident two-pixel displacement right up to the grid boundary.
    # The former abrupt support cutoff collapsed adjacent columns there.
    matcher._cuda_costs = lambda image: np.broadcast_to(peak, (len(matcher.templates), 7, 7))
    displacement = matcher.displacement(ref, (.375, -.625))
    for value, expected in zip(displacement, (.375, -.625)):
        np.testing.assert_array_equal(value, np.full(ref.shape, expected))
