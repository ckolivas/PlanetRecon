"""Capture screening: physical distortions, selection, and two-pass wiring."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from planetrecon.detector import cfa_labels
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.preprocess import measure_frame, screen_source, sigma_selection
from planetrecon.reconstruction import ReconstructionConfig


def planet(rx=23, ry=18, angle=0, phase=False, rings=False, blur=.7):
    y, x = np.indices((96, 112), dtype=float)
    x -= 55.5
    y -= 47.5
    u = np.cos(angle)*x + np.sin(angle)*y
    v = -np.sin(angle)*x + np.cos(angle)*y
    disc = (u/rx)**2 + (v/ry)**2 < 1
    if phase:
        disc &= u > -.25 * rx
    image = disc * (100 + 15*np.cos(v*1.2) + 5*np.cos(u*.7))
    if rings:
        radius = (u/39)**2 + (v/10)**2
        image = np.maximum(image, 70*((radius < 1) & (radius > .65)))
    return 10 + gaussian_filter(image, blur)


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, batch_frames=5, **kwargs)


def test_exact_simultaneous_two_sigma_rules_and_degenerate_population():
    values = np.tile([10., 40., 30.], (30, 1))
    values[0, 0] = 0
    values[1, 0] = 20  # high quality is not rejected
    values[2, 1] = 60
    values[3, 1] = 20
    values[4, 2] = 45
    values[5, 2] = 15
    keep, reasons, stats = sigma_selection(values, np.ones(30, bool))
    assert np.flatnonzero(~keep).tolist() == [0, 2, 3, 4, 5]
    assert stats['quality']['std'] == pytest.approx(np.std(values[:, 0]))
    assert stats['quality']['upper'] is None
    # A value exactly two population standard deviations below the mean stays.
    boundary = np.tile([10., 40., 30.], (5, 1))
    boundary[0, 0] = 0
    assert sigma_selection(boundary, np.ones(5, bool))[0].all()
    for n in (1, 12):
        assert sigma_selection(np.ones((n, 3)), np.ones(n, bool))[0].all()


@pytest.mark.parametrize('phase,rings', [(False, False), (True, False), (False, True)])
def test_silhouette_dimensions_follow_roll_and_do_not_force_a_disc(phase, rings):
    dims = []
    for angle in (0, .4, .9, 1.5):
        q, width, height, pa, status = measure_frame(planet(angle=angle, phase=phase, rings=rings), 'mono')
        assert q > 0 and status == 'ok'
        dims.append((width, height))
    np.testing.assert_allclose(dims, np.broadcast_to(dims[0], (4, 2)), atol=3)
    if rings:
        assert dims[0][0] > 70 and dims[0][1] < 45


def test_blur_missing_target_and_width_height_outliers_rejected():
    frames = np.stack([planet() for _ in range(40)] +
                      [planet(blur=7), planet(rx=33), planet(ry=30), np.zeros((96, 112))])
    selection = screen_source(ArraySource(frames), config())
    assert selection.accepted[:40].all()
    assert not selection.accepted[40:].any()
    reasons = selection.summary['rejected_indices_by_reason']
    assert 40 in reasons['low_quality']
    assert 41 in reasons['width_outlier']
    assert 42 in reasons['height_outlier'] or 42 in reasons['width_outlier']
    assert reasons['no_target'] == [43]


@pytest.mark.parametrize('pattern', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
def test_bayer_colour_lattice_does_not_outrank_sharpness(pattern):
    labels = cfa_labels(96, 112, pattern)
    response = np.where(labels == 'R', .6, np.where(labels == 'B', .3, 1.))
    sharp = measure_frame(planet()*response, pattern)
    blurred = measure_frame(planet(blur=5)*response, pattern)
    assert sharp[-1] == blurred[-1] == 'ok'
    assert sharp[0] > blurred[0] * 2


@pytest.mark.parametrize('mode', ['none', 'field'])
def test_processing_only_accumulates_selected_frames_after_full_scan(mode):
    frames = np.stack([planet() for _ in range(30)] + [planet(blur=7)])
    class TrackedSource(ArraySource):
        def __init__(self):
            super().__init__(frames)
            self.reads = []
        def read_raw(self, index):
            self.reads.append(index)
            return super().read_raw(index)
    source = TrackedSource()
    stages = []
    def event(result, info):
        stages.append(result.stage)
        if result.n_used:
            assert set(range(len(frames))).issubset(source.reads)
    cfg = config(geometry_mode=mode, field_rate_rad_s=0 if mode == 'field' else None)
    result = stack_source(source, cfg, on_event=event)
    assert result.n_used == 30 and result.n_rejected == 1
    assert result.provenance['preprocessing']['n_accepted'] == 30
    assert stages[0] == 'preprocessing' and stages[-1] == 'final'
    expected = stack_source(ArraySource(frames[:30]), replace(cfg, frame_preselection=False))
    np.testing.assert_allclose(result.image, expected.image, atol=1e-10)


def test_cancel_during_screening_never_starts_processing(tmp_path):
    stop = False
    def progress(result, info):
        nonlocal stop
        if info['n_processed'] >= 5:
            stop = True
    destination = tmp_path / 'state.npz'
    result = stack_source(ArraySource(np.stack([planet()]*20)), config(),
                          should_cancel=lambda: stop, on_event=progress, state_checkpoint=destination)
    assert result.incomplete and result.stage == 'preprocessing' and result.n_used == 0
    assert not destination.exists()


def test_rejected_explicit_reference_and_all_rejected_fail_clearly():
    frames = np.stack([planet()]*30 + [planet(blur=7)])
    with pytest.raises(ValueError, match='reference frame was rejected'):
        stack_source(ArraySource(frames), config(reference_index=30))
    with pytest.raises(ValueError, match='no usable frames remain after preprocessing'):
        stack_source(ArraySource(np.zeros((3, 32, 32))), config())


def test_preprocessing_config_rejects_non_boolean():
    with pytest.raises(ValueError, match='frame_preselection'):
        config(frame_preselection='false')


def test_clipped_target_and_calibration_are_measured_before_statistics():
    from planetrecon.calibration import Calibration
    frames = np.stack([planet()]*30 + [np.roll(planet(), 45, axis=1)])
    baseline = screen_source(ArraySource(frames), config())
    calibrated = screen_source(ArraySource(frames + 100), config(),
                               Calibration(bias=np.full(frames.shape[1:], 100.), gain_e_per_adu=2))
    assert baseline.summary['rejected_indices_by_reason']['clipped_target'] == [30]
    np.testing.assert_array_equal(baseline.accepted, calibrated.accepted)
    np.testing.assert_allclose(calibrated.measurements[:30, 0], baseline.measurements[:30, 0]*4)
    np.testing.assert_allclose(calibrated.measurements[:30, 1:], baseline.measurements[:30, 1:])


def test_screened_translation_resume_retains_indices_and_counts(tmp_path):
    from planetrecon.io.ser import write_ser, SERSource
    frames = np.stack([planet()]*30 + [planet(blur=7)]).astype('u2')
    frames[2] = 65535
    path = write_ser(tmp_path/'capture.ser', frames)
    cfg = config()
    state = tmp_path/'state.npz'
    cancel = False
    def event(result, info):
        nonlocal cancel
        if result.stage != 'preprocessing' and info['n_processed'] >= 10:
            cancel = True
    with SERSource(path) as source:
        full = stack_source(source, cfg)
        partial = stack_source(source, cfg, state_checkpoint=state,
                               should_cancel=lambda: cancel, on_event=event)
        resumed = stack_source(source, cfg, resume_from=state)
    assert partial.n_used == 9 and partial.n_rejected == 1
    assert full.n_used == resumed.n_used == 29
    assert full.n_rejected == resumed.n_rejected == 2
    np.testing.assert_array_equal(full.image, resumed.image)
    np.testing.assert_array_equal(full.coverage, resumed.coverage)
    assert full.provenance['preprocessing'] == resumed.provenance['preprocessing']


def test_worker_reports_screening_without_blank_previews(tmp_path):
    import time
    from planetrecon.io.ser import write_ser
    from planetrecon.jobs import start_stack_job
    path = write_ser(tmp_path/'capture.ser', np.stack([planet()]*6).astype('u2'))
    handle = start_stack_job(path, config())
    events = []
    try:
        deadline = time.monotonic() + 15
        while handle.state not in ('failed', 'completed') and time.monotonic() < deadline:
            events.extend(handle.poll(.05))
        assert handle.state == 'completed'
        assert any(e.kind == 'progress' and e.payload.get('stage', '').startswith('preprocessing') for e in events)
        assert not any(e.kind == 'preview' and e.payload.get('stage') == 'preprocessing' for e in events)
    finally:
        handle.close()
