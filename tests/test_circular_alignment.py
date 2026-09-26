"""Scientific invariants and workflow tests for sampling-sized classic stacking."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from planetrecon.pipeline.circular_align import CircularMultiscaleRegistration, minimum_diameter
from planetrecon.pipeline.local_align import pull
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess_cache import preprocess_source
from planetrecon.io.ser import SERSource
from planetrecon.reconstruction import ReconstructionConfig
from test_local_alignment import scene, capture, config


def circular_config(**kwargs):
    return config(alignment_method='circular_multiscale', **kwargs)


@pytest.mark.parametrize('multiple,wavelength,expected', [(5, 550, 23), (7, 550, 33),
                                                        (7, 450, 27), (5, 850, 35)])
def test_sampling_sets_minimum_and_overlapping_scales(multiple, wavelength, expected):
    matcher = CircularMultiscaleRegistration(scene(), sampling_multiplier=multiple, wavelength_nm=wavelength)
    assert minimum_diameter(multiple, wavelength) == expected
    assert matcher.layers[0].window == expected
    assert all(layer.step <= layer.window/2 for layer in matcher.layers)
    for layer in matcher.layers:
        y, x = np.indices(layer.weight.shape) - layer.window//2
        assert np.all(layer.weight[np.hypot(x, y) >= layer.window//2] == 0)
        np.testing.assert_array_equal(layer.weight, layer.weight.T)
        assert layer.weight.sum() == pytest.approx(1.)


@pytest.mark.parametrize('kwargs', [dict(sampling_multiplier=v) for v in (None, True, 0, 21, float('nan'))]
                         + [dict(alignment_wavelength_nm=v) for v in (None, True, 299, float('inf'))])
def test_invalid_sampling_refused(kwargs):
    with pytest.raises(ValueError):
        ReconstructionConfig(**kwargs)


def test_circular_method_cannot_silently_use_square_motion_path():
    with pytest.raises(ValueError, match='Motion model None'):
        circular_config(geometry_mode='surface')


def test_local_distortion_recovered_without_sharpening_or_folds():
    ref = scene()
    y, x = np.indices(ref.shape)
    flow = 1.5*np.sin(2*np.pi*y/ref.shape[0])
    frame = map_coordinates(ref, [y, x-flow], order=3, mode='reflect')
    matcher = CircularMultiscaleRegistration(ref)
    d = matcher.displacement(frame, (0., 0.))
    corrected = pull(frame, d)
    roi = np.s_[35:-35, 35:-35]
    assert np.linalg.norm((corrected-ref)[roi]) < .25*np.linalg.norm((frame-ref)[roi])
    assert corrected[roi].min() >= frame.min() and corrected[roi].max() <= frame.max()
    jac = ((1+np.gradient(d[0], axis=1))*(1+np.gradient(d[1], axis=0))
           - np.gradient(d[0], axis=0)*np.gradient(d[1], axis=1))
    assert jac.min() >= .25
    # Fine layers actually improve the same coarse-only estimate.
    matcher.layers = matcher.layers[-1:]
    coarse = pull(frame, matcher.displacement(frame, (0., 0.)))
    assert np.linalg.norm((corrected-ref)[roi]) < .8*np.linalg.norm((coarse-ref)[roi])


def test_flat_ambiguous_unrelated_and_small_regions_fall_back():
    y, x = np.indices((100, 120))
    for ref, frame in [(np.ones((100, 120)), np.ones((100, 120))),
                       (300+100*np.sin((x+y)/4), 300+100*np.sin((x+y-1.3)/4)),
                       (np.ones((20, 24)), np.ones((20, 24)))]:
        matcher = CircularMultiscaleRegistration(ref)
        d = matcher.displacement(frame, (.3, -.2))
        for axis, value in zip(d, (.3, -.2)):
            np.testing.assert_allclose(axis, value, atol=1e-15)
    ref = scene()
    other = gaussian_filter(np.random.default_rng(921).normal(size=ref.shape), 1.5)*100+300
    d = CircularMultiscaleRegistration(ref).displacement(other, (0., 0.))
    assert np.max(np.abs(d)) == 0


def test_brightness_and_blur_do_not_invent_material_motion():
    ref = scene()
    matcher = CircularMultiscaleRegistration(ref)
    assert np.max(np.abs(matcher.displacement(ref, (0, 0)))) == 0
    assert np.max(np.abs(matcher.displacement(.8*ref+25, (0., 0.)))) == 0
    blur = matcher.displacement(gaussian_filter(ref, 1.), (0., 0.))
    assert np.sqrt(np.mean(np.asarray(blur)[:, 35:-35, 35:-35]**2)) < .05


def test_detector_edges_retain_global_motion():
    ref = scene()
    y, x = np.indices(ref.shape)
    frame = map_coordinates(ref, [y-.5, x-1.2], order=3, mode='constant')
    global_shift = (1., .4)
    d = CircularMultiscaleRegistration(ref).displacement(frame, global_shift)
    for a, expected in zip(d, global_shift):
        np.testing.assert_allclose(a[:3], expected)
        np.testing.assert_allclose(a[-3:], expected)
        np.testing.assert_allclose(a[:, :3], expected)
        np.testing.assert_allclose(a[:, -3:], expected)


def test_cancel_between_scales():
    matcher = CircularMultiscaleRegistration(scene())
    matcher.should_cancel = lambda: True
    with pytest.raises(InterruptedError, match='circular alignment'):
        matcher.displacement(scene(), (0., 0.))


def test_unreliable_small_samples_preserve_coarse_solution():
    ref = scene()
    y, x = np.indices(ref.shape)
    frame = map_coordinates(ref, [y, x-.8*np.sin(y/25)], order=3, mode='reflect')
    matcher = CircularMultiscaleRegistration(ref)
    for layer in matcher.layers[:-1]:
        layer.texture_valid = [False]*len(layer.texture_valid)
    fallback = matcher.displacement(frame, (0., 0.))
    matcher.layers = matcher.layers[-1:]
    coarse = matcher.displacement(frame, (0., 0.))
    np.testing.assert_array_equal(fallback, coarse)
    assert np.max(np.abs(coarse)) > .1


def test_cli_circular_inputs_reach_result(tmp_path):
    from planetrecon.cli import main
    from planetrecon.result import load_snapshot
    path = capture(tmp_path)
    with SERSource(path) as source:
        preprocess_source(source, config())
    assert main(['--threads', '2', 'stack', '--path', str(path), '--device', 'cpu',
                 '--alignment-method', 'circular_multiscale', '--sampling-multiplier', '7',
                 '--alignment-wavelength-nm', '450', '--stack-percent', '100',
                 '--out', str(tmp_path/'result')]) == 0
    detail = load_snapshot(tmp_path/'result/stack.npz').provenance['local_alignment']
    assert detail['method'] == 'circular_multiscale' and detail['minimum_diameter_px'] == 27


@pytest.mark.parametrize('color_id', [0, 8, 100])
def test_stack_reuses_cache_and_preserves_raw_colour_support(tmp_path, color_id):
    with SERSource(capture(tmp_path, color_id)) as source:
        cfg = circular_config()
        selection = preprocess_source(source, cfg)
        cache = source.path.with_name(source.path.name+'.planetrecon-preprocess.npz')
        original = cache.read_bytes()
        result = stack_source(source, cfg)
        assert result.n_used == selection.accepted.sum()
        assert not result.incomplete and np.isfinite(result.image).all()
        assert result.channel_order == ('mono' if color_id == 0 else 'RGB')
        assert result.validity[30:65, 35:75].all()
        assert result.provenance['local_alignment']['diameters_px'] == [23, 33, 47, 67]
        changed = stack_source(source, replace(cfg, sampling_multiplier=7))
        assert changed.provenance['local_alignment']['minimum_diameter_px'] == 33
        assert changed.n_used == result.n_used
        assert cache.read_bytes() == original
        assert not np.array_equal(changed.image, result.image)


def test_checkpoint_resume_exact_and_sampling_bound(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        cfg = circular_config()
        preprocess_source(source, cfg)
        whole = stack_source(source, cfg)
        stop = False
        def event(result, info):
            nonlocal stop
            stop = info['n_processed'] >= 4
        checkpoint = tmp_path/'circular.npz'
        partial = stack_source(source, cfg, state_checkpoint=checkpoint,
                               on_event=event, should_cancel=lambda: stop)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, whole.image)
        np.testing.assert_array_equal(resumed.coverage, whole.coverage)
        for changed in (replace(cfg, sampling_multiplier=7), replace(cfg, alignment_wavelength_nm=650),
                        replace(cfg, alignment_method='square')):
            with pytest.raises(ValueError, match='identity'):
                stack_source(source, changed, resume_from=checkpoint)


def test_oversized_sampling_uses_explicit_global_fallback(tmp_path):
    with SERSource(capture(tmp_path)) as source:
        cfg = circular_config(sampling_multiplier=20, alignment_wavelength_nm=1500)
        preprocess_source(source, cfg)
        result = stack_source(source, cfg)
        assert result.n_used > 0 and not result.incomplete
        assert result.provenance['local_alignment']['diameters_px'] == []
        assert any('using global alignment' in warning for warning in result.warnings)


def test_stack_preserves_absolute_frame_brightness(tmp_path):
    """Matching normalisation must never reach the accumulated observations."""
    from planetrecon.io.ser import write_ser
    with SERSource(capture(tmp_path)) as original:
        base = original.read_raw(0).astype(float)
    frames = np.stack([base*gain + 31*i for i, gain in enumerate(range(1, 9))]).astype('u2')
    path = write_ser(tmp_path/'brightness.ser', frames)
    with SERSource(path) as source:
        cfg = circular_config()
        selected = preprocess_source(source, cfg)
        result = stack_source(source, cfg)
    kept = selected.accepted
    weights = np.maximum(selected.measurements[kept, 0], 1e-12)
    expected = np.average(frames[kept].astype(float), axis=0, weights=weights)
    assert result.n_used == kept.sum() >= 4
    # Gain and background offsets both survive as recorded, in ADU. A common
    # reference exposure/mean normalisation would fail this pixelwise test.
    np.testing.assert_allclose(result.image[20:-20, 20:-20], expected[20:-20, 20:-20], rtol=1e-8, atol=1e-6)
    assert result.units.lower() == 'adu'
    assert 'no per-frame brightness normalisation' in result.provenance['frame_brightness']


def test_capture_controls_persist_sampling_and_classic_stack(tmp_path):
    from planetrecon.gui.app import MainWindow, create_app
    from test_w14 import pump
    app = create_app([])
    settings_path = tmp_path/'settings.json'
    win = MainWindow(config=ReconstructionConfig(device='cpu', threads=2), settings_path=settings_path)
    try:
        assert not win.stack_btn.isEnabled() and win.path is None
        fields = win.controls.fields
        assert fields['sampling_multiplier'].text() == '5.0'
        assert '23 px' in win.controls.alignment_size_label.text()
        fields['sampling_multiplier'].setText('7')
        assert '33 px' in win.controls.alignment_size_label.text()
        fields['alignment_wavelength_nm'].setText('450')
        assert '27 px' in win.controls.alignment_size_label.text()
        # Classic Stack must not demand the incomplete motion geometry.
        fields['geometry_mode'].setCurrentIndex(fields['geometry_mode'].findData('saturn'))
        win.path = capture(tmp_path)
        win._buttons()
        win.stack_btn.click()
        assert not win.stack_btn.isEnabled()
        pump(app, lambda: win.job is None and win.run_stage is None, timeout=30)
        assert not win.error.text()
        assert win.stack_btn.isEnabled() and win.last_result.n_used > 0
        detail = win.last_result.provenance['local_alignment']
        assert detail['method'] == 'circular_multiscale'
        assert detail['minimum_diameter_px'] == 27
        assert win.config.geometry_mode == 'none'
        assert not fields['local_patch_size'].isEnabled()
    finally:
        win.window.close()
    restored = MainWindow(settings_path=settings_path)
    try:
        assert restored.path is None and restored.job is None
        assert not restored.stack_btn.isEnabled()
        assert restored.controls.configuration().sampling_multiplier == 7
        assert '27 px' in restored.controls.alignment_size_label.text()
    finally:
        restored.window.close()


@pytest.mark.hardware
def test_cuda_matches_cpu():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    ref = scene()
    y, x = np.indices(ref.shape)
    frame = map_coordinates(ref, [y+.25, x-.8*np.sin(y/25)], order=3, mode='reflect')
    cpu = CircularMultiscaleRegistration(ref).displacement(frame, (0., 0.))
    gpu = CircularMultiscaleRegistration(ref, use_cuda=True).displacement(frame, (0., 0.))
    np.testing.assert_allclose(cpu, gpu, atol=1e-9, rtol=1e-9)
