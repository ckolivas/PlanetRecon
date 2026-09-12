"""Unavailable selected compensation must never silently become an ordinary stack."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.geometry_stack import prepare_geometry
from planetrecon.reconstruction import ReconstructionConfig
from test_globe_registration import config, sphere


@pytest.mark.parametrize('mode', ['surface', 'combined', 'saturn'])
def test_unresolved_surface_rate_refuses_before_emitting_output(mode, tmp_path):
    src = ArraySource(np.array([sphere(0), sphere(1)]), bit_depth=32, timestamps=np.arange(2.))
    cfg = replace(config(), geometry_mode=mode, surface_rate_rad_s=None,
                  sub_obs_lat_rad=.4 if mode == 'saturn' else None,
                  ring_inner_radius_px=50 if mode == 'saturn' else None,
                  ring_outer_radius_px=60 if mode == 'saturn' else None)
    events = []
    checkpoint = tmp_path/'state.npz'
    checkpoint.write_bytes(b'previous state')
    with pytest.raises(ValueError, match='Required: Surface rotation rate'):
        stack_source(src, cfg, state_checkpoint=checkpoint,
                     on_event=lambda *args: events.append(args))
    assert not events
    assert checkpoint.read_bytes() == b'previous state'


def test_explicit_zero_is_distinct_from_failed_field_estimate():
    disc = np.ones((64, 64))
    src = ArraySource(np.stack([disc]*3), bit_depth=32, timestamps=np.arange(3.))
    cfg = ReconstructionConfig(device='cpu', threads=2, geometry_mode='field',
        field_center_x=32, field_center_y=32, equatorial_radius_px=20)
    with pytest.raises(ValueError, match='field rotation could not be estimated'):
        stack_source(src, cfg)
    result = stack_source(src, replace(cfg, field_rate_rad_s=0.))
    assert result.n_used == 3
    np.testing.assert_allclose(result.image, disc, atol=1e-12)
    # Geometry inspection still returns diagnostics for an unresolved estimate.
    _, _, diagnostics, _ = prepare_geometry(src, cfg)
    assert diagnostics['unavailable_motion']


def test_unresolved_surface_tracking_refuses_before_preview_or_checkpoint(tmp_path):
    src = ArraySource(np.ones((3,128,128)), bit_depth=32, timestamps=np.arange(3.))
    events = []
    checkpoint = tmp_path/'state.npz'
    checkpoint.write_bytes(b'previous state')
    with pytest.raises(ValueError, match='shared visible surface cannot constrain camera drift'):
        stack_source(src, config(), on_event=lambda *args: events.append(args),
                     state_checkpoint=checkpoint)
    assert not events and checkpoint.read_bytes() == b'previous state'


def test_later_tracking_failure_never_returns_a_completed_stack():
    frames = np.array([sphere(i) for i in range(5)])
    frames[1] = 1.  # Not one of the first/middle/last estimation frames.
    src = ArraySource(frames, bit_depth=32, timestamps=np.arange(5.))
    events = []
    with pytest.raises(ValueError, match='frame 2'):
        stack_source(src, config(), on_event=lambda result, info: events.append(result))
    assert all(result.incomplete for result in events)


def test_unresolved_saturn_ring_translation_cannot_supply_a_field_rate(monkeypatch):
    from test_ring_registration import source as ring_source, config as ring_config
    from planetrecon.pipeline.ring_align import RingRegistration
    # Polar texture alone must not certify a rate when camera registration fails.
    monkeypatch.setattr(RingRegistration, 'displacement', lambda *args, **kwargs: None)
    src = ring_source('mono', [(0, 0)]*5, field_rate=.03)
    cfg = replace(ring_config(), field_rate_rad_s=None)
    _, _, diagnostics, _ = prepare_geometry(src, cfg)
    assert diagnostics['field_sample_shifts_px'] == [[0., 0.], None, None]
    assert diagnostics['unavailable_motion']
    with pytest.raises(ValueError, match='field rotation could not be estimated'):
        stack_source(src, cfg)


@pytest.mark.parametrize('note', [
    'Rings or strong phase: supply the globe radius to estimate surface rotation.',
    'Surface rotation is unresolved; no surface-rate prefill.',
    'Saturn needs a supplied signed viewing latitude.',
])
def test_missing_rate_explains_the_preprocessing_blocker(note):
    cfg = ReconstructionConfig(geometry_mode='surface')
    info = {'status': 'ready', 'geometry_estimate': {
        'status': 'unresolved', 'suggestions': {}, 'notes': [note]}}
    with pytest.raises(ValueError) as caught:
        cfg.require_motion_parameters(info)
    assert note in str(caught.value)
    assert 'latest preprocessing did not supply a usable surface rate' in str(caught.value)


@pytest.mark.parametrize('info', [
    {'status': 'stale', 'geometry_estimate': {'notes': ['old reason']}},
    {'status': 'ready', 'geometry_estimate': {'applicable': False, 'notes': ['old reason']}},
])
def test_unverified_preprocessing_is_not_presented_as_a_current_failure(info):
    with pytest.raises(ValueError) as caught:
        ReconstructionConfig(geometry_mode='surface').require_motion_parameters(info)
    assert 'old reason' not in str(caught.value)
    assert 'Preprocess may estimate' in str(caught.value)


def test_estimated_but_unapplied_rate_gets_distinct_guidance():
    info = {'status': 'ready', 'geometry_estimate': {
        'suggestions': {'surface_rate_rad_s': .001}, 'notes': []}}
    with pytest.raises(ValueError, match='estimated a surface rate, but it is not set'):
        ReconstructionConfig(geometry_mode='surface').require_motion_parameters(info)
    # Diagnostics never silently change the selected rate or weaken refusal.
    ReconstructionConfig(geometry_mode='surface', surface_rate_rad_s=.001).require_motion_parameters(info)


def test_gui_stack_shows_saturn_preprocessing_reason_after_preparation(monkeypatch):
    from pathlib import Path
    from planetrecon.gui.app import MainWindow, create_app
    import planetrecon.gui.app as gui
    app = create_app(['motion-preprocess-reason'])
    win = MainWindow(config=ReconstructionConfig(device='cpu', geometry_mode='saturn'))
    win.path = Path('saturn-l3.ser')
    # Explicit opt-out keeps this test on the unresolved measured-rate path.
    win.controls.fields['rotation_planet'].activated.emit(0)
    note = 'Rings or strong phase: supply the globe radius to estimate surface rotation.'
    win.preprocessing_info = {'status': 'ready', 'geometry_estimate': {
        'status': 'unresolved', 'suggestions': {}, 'notes': [note]}}

    def unexpected(*args, **kwargs):
        pytest.fail('an unavailable motion model must not start a worker')

    monkeypatch.setattr(gui, 'start_stack_job', unexpected)
    try:
        win.run_stage = 'preprocess'
        win._advance_run('completed')
        assert win.job is None and note in win.error.text()
        assert 'Globe equatorial radius' in win.error.text()
        assert 'Signed observer latitude' not in win.error.text()
    finally:
        win._shutdown()
        win.window.close()
        app.processEvents()
