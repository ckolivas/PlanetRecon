"""Stack keeps the selected motion and a reference independent of stack cutoff."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.motion_reference import motion_reference_plan
from planetrecon.pipeline.preprocess import FrameSelection
from planetrecon.reconstruction import ReconstructionConfig


def test_reference_uses_elapsed_time_quality_and_screening():
    times = np.r_[np.linspace(0, 1, 80), 4., 4.5, 5., 5.5, 6., 9., 10.]
    source = ArraySource(np.zeros((len(times), 8, 8)), timestamps=times)
    scores = np.ones((len(times), 4))
    scores[:, 0] = np.arange(len(times)) + 1
    scores[0, 0] = 1000  # Best overall, outside the central time window.
    kept = np.ones(len(times), dtype=bool)
    kept[82] = False  # Exactly at midpoint, but screened out.
    selection = FrameSelection(kept, scores, {})
    plan = motion_reference_plan(source, ReconstructionConfig(), selection)
    assert plan['anchor_index'] == 83
    assert plan['target_time'] == 5.
    assert set(plan['template_candidates']) == {80, 81, 83, 84}
    assert plan['time_basis'] == 'seconds from start'
    best = motion_reference_plan(source, ReconstructionConfig(motion_reference='best'), selection)
    assert best['anchor_index'] == 0
    manual = motion_reference_plan(source, ReconstructionConfig(reference_index=81), selection)
    assert manual['anchor_index'] == 81 and manual['policy'] == 'manual'


def test_reference_without_timestamps_and_sparse_screening():
    source = ArraySource(np.zeros((101, 8, 8)))
    kept = np.zeros(101, bool)
    kept[[0, 2, 90, 99]] = True
    plan = motion_reference_plan(source, ReconstructionConfig(),
                                 FrameSelection(kept, np.ones((101, 4)), {}))
    assert plan['anchor_index'] == 90
    assert set(plan['template_candidates']) == {0, 2, 90, 99}
    assert plan['time_basis'] == 'frame index'


@pytest.mark.parametrize('mode', ['field', 'surface', 'combined', 'saturn'])
@pytest.mark.parametrize('method', ['square', 'circular_multiscale'])
def test_stack_motion_reference_cutoff_and_resume(tmp_path, mode, method):
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.preprocess_cache import preprocess_source
    from test_best_reference import capture
    cfg = ReconstructionConfig(device='cpu', threads=2, batch_frames=2, geometry_mode=mode,
        cadence_s=.1, reference_epoch_s=.75, field_rate_rad_s=0., surface_rate_rad_s=0.,
        field_center_x=40.5, field_center_y=32.5, equatorial_radius_px=21.,
        sub_obs_lat_rad=.4, ring_inner_radius_px=25. if mode == 'saturn' else None,
        ring_outer_radius_px=30. if mode == 'saturn' else None,
        local_alignment=True, local_patch_size=33, alignment_method=method,
        quality_weighting=False, frame_selection_mode='frame_count', stack_percent=25)
    with SERSource(capture(tmp_path)) as source:
        selection = preprocess_source(source, cfg)
        result = stack_source(source, cfg)
        anchor = result.provenance['reference_index']
        assert anchor == selection.summary['geometry_estimate']['reference_index'] == 7
        assert result.provenance['motion_reference']['anchor_index'] == anchor
        assert result.n_used == 4  # Anchor/template frames are not extra stack votes.
        assert float(result.reference_epoch) == .75
        assert result.provenance['local_alignment']['method'] == method
        assert 'no per-frame brightness normalisation' in result.provenance['frame_brightness']
        # Brightness doubled in one selected frame; the remaining three are normal.
        expected = source.read_raw(0).astype(float) * 1.25
        np.testing.assert_allclose(result.image[25:39, 33:47], expected[25:39, 33:47], atol=1e-7)
        stop = False
        def event(_, info):
            nonlocal stop
            stop = info.get('n_processed', 0) >= 2
        checkpoint = tmp_path/'motion.npz'
        partial = stack_source(source, cfg, state_checkpoint=checkpoint,
                               should_cancel=lambda: stop, on_event=event)
        assert partial.incomplete
        resumed = stack_source(source, cfg, resume_from=checkpoint)
        np.testing.assert_array_equal(resumed.image, result.image)
        np.testing.assert_array_equal(resumed.coverage, result.coverage)
        single = stack_source(source, replace(cfg, stack_percent=1))
        assert single.n_used == 1
        assert single.provenance['reference_index'] == anchor
        with pytest.raises(ValueError, match='identity/configuration mismatch'):
            stack_source(source, replace(cfg, motion_reference='best'), resume_from=checkpoint)


@pytest.mark.parametrize('method', ['square', 'circular_multiscale'])
def test_rotating_surface_improves_against_independent_midpoint_truth(method):
    from test_globe_registration import sphere, config
    from test_motion_local import selected
    rate = .04
    times = np.arange(9.)
    frames = np.array([sphere(t, rate=rate) for t in times])
    source = ArraySource(frames, bit_depth=32, timestamps=times)
    cfg = replace(config(rate=rate), frame_preselection=True, local_alignment=True, motion_reference='midpoint',
                  local_patch_size=33, alignment_method=method, reference_epoch_s=4.,
                  quality_weighting=False)
    selection = selected(source, cfg)
    selection.measurements[:, 0] = 1.
    from planetrecon.pipeline.preprocess_cache import selection_digest
    selection.digest = selection_digest(selection)
    corrected = stack_source(source, cfg, preprocessing=selection)
    ordinary = stack_source(source, replace(cfg, geometry_mode='none', reference_index=4),
                            preprocessing=selection)
    truth = sphere(4., rate=rate)
    y, x = np.indices(truth.shape)
    roi = (x+.5-64)**2 + (y+.5-64)**2 < 25**2
    corrected_error = np.sqrt(np.mean((corrected.image[roi]-truth[roi])**2))
    ordinary_error = np.sqrt(np.mean((ordinary.image[roi]-truth[roi])**2))
    assert corrected.n_used == ordinary.n_used == 9
    assert corrected.provenance['reference_index'] == 4
    assert corrected_error < .6 * ordinary_error


@pytest.mark.parametrize('mode', ['field', 'surface', 'combined', 'saturn'])
def test_stack_button_preserves_motion_and_circular_choice(monkeypatch, tmp_path, mode):
    from planetrecon.gui.app import MainWindow, create_app
    app = create_app([])
    window = MainWindow(config=ReconstructionConfig(device='cpu', geometry_mode=mode,
                                                    alignment_method='circular_multiscale'))
    window.path = tmp_path/'capture.ser'
    calls = []
    monkeypatch.setattr(window, '_run', lambda: calls.append(window.controls.configuration()))
    try:
        window._stack()
        assert len(calls) == 1
        assert calls[0].geometry_mode == mode
        assert calls[0].alignment_method == 'circular_multiscale'
        assert calls[0].motion_reference == 'midpoint'
        assert calls[0].local_alignment and calls[0].frame_preselection
        assert window.controls.fields['alignment_method'].isEnabled()
    finally:
        window.window.close()
        app.processEvents()


@pytest.mark.parametrize('method', ['square', 'circular_multiscale'])
def test_unobserved_motion_template_retains_only_model_motion(method):
    from planetrecon.pipeline.motion_local import local_coordinates
    rng = np.random.default_rng(703)
    reference = rng.normal(size=(80, 96))
    prepared = np.stack((reference, np.zeros_like(reference)), axis=-1)
    x, y = local_coordinates(prepared, np.roll(reference, 2, axis=1), None, None,
                              lambda image, *args: image, 33, False, method=method)
    yy, xx = np.indices(reference.shape)
    np.testing.assert_array_equal(x, xx+.5)
    np.testing.assert_array_equal(y, yy+.5)


def test_old_geometry_refreshes_without_invalidating_quality(tmp_path):
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.preprocess_cache import preprocess_source, cache_report
    from test_best_reference import capture
    cfg = ReconstructionConfig(device='cpu', threads=2, cadence_s=.1)
    with SERSource(capture(tmp_path)) as source:
        selection = preprocess_source(source, cfg)
        kept, quality = selection.accepted.copy(), selection.measurements.copy()
        assert cache_report(selection, config=cfg)['geometry_estimate'].get('applicable', True)
        changed = cache_report(selection, config=replace(cfg, motion_reference='best'))
        assert changed['status'] == 'ready' and not changed['geometry_estimate']['applicable']
        del selection.summary['geometry_analysis_config']['motion_reference']
        report = cache_report(selection, config=cfg)
        assert report['status'] == 'ready' and not report['geometry_estimate']['applicable']
        assert not report['geometry_estimate']['suggestions']
        np.testing.assert_array_equal(selection.accepted, kept)
        np.testing.assert_array_equal(selection.measurements, quality)


def test_uncached_motion_skips_invalid_midpoint_but_honours_manual_reference():
    from test_globe_registration import sphere, config
    frames = np.stack([sphere(0.)]*5)
    frames[2] = np.nan
    source = ArraySource(frames, bit_depth=32, timestamps=np.arange(5.))
    cfg = replace(config(rate=0.), motion_reference='midpoint')
    result = stack_source(source, cfg)
    assert result.n_used == 4 and result.n_rejected == 1
    assert result.provenance['reference_index'] == 1
    with pytest.raises(ValueError, match='reference frame'):
        stack_source(source, replace(cfg, reference_index=2))


def test_invalid_manual_anchor_does_not_discard_preprocessing(tmp_path):
    from planetrecon.io.ser import SERSource
    from planetrecon.pipeline.preprocess_cache import preprocess_source
    from test_best_reference import capture
    cfg = ReconstructionConfig(device='cpu', threads=2, cadence_s=.1, reference_index=100)
    with SERSource(capture(tmp_path)) as source:
        selection = preprocess_source(source, cfg)
        assert selection.accepted.sum() == 16
        assert 'Motion reference is unavailable' in ' '.join(selection.summary['geometry_estimate']['notes'])
        with pytest.raises(ValueError, match='reference_index is outside'):
            stack_source(source, cfg)


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_circular_motion_cuda_matches_cpu(colour):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_cuda_saturn import capture, equal_result
    from test_motion_local import selected
    source, cfg = capture(colour)
    cfg = replace(cfg, frame_preselection=True, local_alignment=True,
                  alignment_method='circular_multiscale', quality_weighting=False)
    selection = selected(source, cfg)
    cpu = stack_source(source, cfg, preprocessing=selection)
    gpu = stack_source(source, replace(cfg, device='gpu'), preprocessing=selection)
    equal_result(gpu, cpu)
