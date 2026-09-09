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
