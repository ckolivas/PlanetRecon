"""Independent colour detail can constrain motion when green alone cannot."""
from dataclasses import replace

import numpy as np
import pytest

from planetrecon.detector import cfa_labels
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig
from test_globe_registration import sphere, config
from test_field_angle_precision import scene


OFFSETS = [(0, 0), (1.3, -2.4), (-3.7, 2.2), (.3, .4), (2.5, -1.2), (0, 0), (-2, 3)]


@pytest.mark.parametrize('pattern', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('detail', ['R', 'B'])
@pytest.mark.parametrize('rate', [-.1, .1])
def test_surface_tracks_red_or_blue_detail(monkeypatch, pattern, detail, rate):
    frames = np.array([sphere(t, rate, dx, dy, .04) for t, (dx, dy) in enumerate(OFFSETS)])
    labels = cfa_labels(128, 128, pattern)
    raw = np.where(labels == detail, frames, np.where(frames > 0, 100., 0.))
    src = ArraySource(raw, color_mode=pattern, bit_depth=32, timestamps=np.arange(7.))
    cfg = config(rate=rate, field_rate=.04)
    corrected = stack_source(src, cfg)
    monkeypatch.setattr('planetrecon.pipeline.globe_align.surface_displacement',
        lambda reference, frame, model, pose, anchor: OFFSETS[round(pose.t_s)])
    oracle = stack_source(src, cfg)
    assert corrected.n_used == oracle.n_used == 7
    assert corrected.n_rejected == oracle.n_rejected == 0
    y, x = np.indices((128, 128))
    mask = (x+.5-64)**2+(y+.5-64)**2 < 25**2
    assert corrected.validity[mask].all()
    assert np.sqrt(np.mean((corrected.image[mask]-oracle.image[mask])**2)) < .15


@pytest.mark.parametrize('pattern', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
def test_automatic_field_rate_can_use_red_detail(pattern):
    rate = np.deg2rad(1)/8
    frames = np.array([scene(rate*t)[0] for t in range(9)])
    labels = cfa_labels(192, 192, pattern)
    raw = np.where(labels == 'R', frames, (frames > 0).astype(float))
    src = ArraySource(raw, color_mode=pattern, bit_depth=32, timestamps=np.arange(9.))
    cfg = ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='field', field_center_x=96, field_center_y=96, equatorial_radius_px=70)
    result = stack_source(src, cfg)
    oracle = stack_source(src, replace(cfg, field_rate_rad_s=rate))
    assert result.n_used == oracle.n_used == 9
    assert result.provenance['geometry']['registration_proxy'] == 'RGB luminance after unresolved green preflight'
    assert result.provenance['geometry']['field_rate_rad_s'] == pytest.approx(rate, rel=.02)
    _, radius = scene(0.)
    mask = (radius > 10) & (radius < 60)
    assert np.sqrt(np.mean((result.image[mask]-oracle.image[mask])**2)) < .001


def test_unresolved_all_colour_motion_emits_no_output(tmp_path):
    # All channels lack interior texture; adding colours cannot invent it.
    src = ArraySource(np.ones((3, 128, 128)), color_mode='RGGB', bit_depth=32,
                      timestamps=np.arange(3.))
    events = []
    checkpoint = tmp_path/'state.npz'
    checkpoint.write_bytes(b'previous checkpoint')
    with pytest.raises(ValueError, match='cannot constrain camera drift'):
        stack_source(src, config(), on_event=lambda *args: events.append(args),
                     state_checkpoint=checkpoint)
    assert not events
    assert checkpoint.read_bytes() == b'previous checkpoint'


@pytest.mark.parametrize('pattern', ['mono', 'RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('t', [3., 6.])
def test_flat_globe_cannot_track_a_stretched_reference_limb(pattern, t):
    from planetrecon.pipeline.geometry_stack import _alignment_plane
    from planetrecon.pipeline.globe_align import surface_displacement
    from planetrecon.geometry.model import OblateGlobeModel
    from planetrecon.geometry.globe import GlobeParams
    from planetrecon.geometry.pose import FramePose
    reference = _alignment_plane(np.where(sphere(0) > 0, 100., 0.), pattern)
    frame = _alignment_plane(np.where(sphere(t, dx=1.3, dy=-2.4) > 0, 100., 0.), pattern)
    model = OblateGlobeModel(GlobeParams(42, surface_rate_rad_s=.1))
    assert surface_displacement(reference, frame, model,
        FramePose(t, 0., 64, 64), FramePose(0., 0., 64, 64)) is None


@pytest.mark.parametrize('index', [1, 3])
@pytest.mark.parametrize('pattern', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
@pytest.mark.parametrize('field_rate', [0., .04])
def test_sampled_and_later_green_failure_track_colour_per_frame(monkeypatch, index, pattern, field_rate):
    frames = np.array([sphere(t, dx=dx, dy=dy, field_rate=field_rate)
                       for t, (dx, dy) in enumerate(OFFSETS)])
    labels = cfa_labels(128, 128, pattern)
    frames[index] = np.where(labels == 'G', np.where(frames[index] > 0, 100., 0.), frames[index])
    src = ArraySource(frames, color_mode=pattern, bit_depth=32, timestamps=np.arange(7.))
    cfg = config(field_rate=field_rate)
    result = stack_source(src, cfg)
    assert 'registration_proxy' not in result.provenance['geometry']
    monkeypatch.setattr('planetrecon.pipeline.globe_align.surface_displacement',
        lambda reference, frame, model, pose, anchor: OFFSETS[round(pose.t_s)])
    oracle = stack_source(src, cfg)
    assert result.n_used == oracle.n_used == 7
    y, x = np.indices((128, 128))
    mask = (x-63.5)**2+(y-63.5)**2 < 25**2
    error = np.sqrt(np.mean((result.image[mask]-oracle.image[mask])**2))
    assert error < .15
    if index == 3 and pattern == 'RGGB':
        # Previously switching every frame to RGB gave 0.02603 / 0.03158.
        assert error < (.030 if field_rate else .024)


def test_colour_retry_resume_and_reference_support_identity(tmp_path):
    import json
    from planetrecon.io.ser import write_ser, SERSource, COLOR_RGGB
    frames = np.array([sphere(t, dx=dx, dy=dy) for t, (dx, dy) in enumerate(OFFSETS)])
    labels = cfa_labels(128, 128, 'RGGB')
    frames = np.where(labels == 'B', frames, np.where(frames > 0, 100., 0.))
    path = write_ser(tmp_path/'in.ser', (frames*100).astype('u2'), color_id=COLOR_RGGB)
    cfg = replace(config(), cadence_s=1., batch_frames=2)
    state = tmp_path/'state.npz'
    with SERSource(path) as src:
        whole = stack_source(src, cfg)
    cancel = False
    def event(result, info):
        nonlocal cancel
        if info['n_processed'] >= 4:
            cancel = True
    with SERSource(path) as src:
        partial = stack_source(src, cfg, state_checkpoint=state, on_event=event,
                               should_cancel=lambda: cancel)
    assert partial.incomplete and partial.n_used == 4
    with SERSource(path) as src:
        continued = stack_source(src, cfg, resume_from=state)
    assert continued.n_used == whole.n_used == 7
    for key in ('image', 'coverage', 'validity'):
        np.testing.assert_array_equal(getattr(continued, key), getattr(whole, key))
    with np.load(state, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    metadata_text = str(arrays['metadata'])
    for previous_policy in ('reference_support', 'colour_tracking_retry'):
        metadata = json.loads(metadata_text)
        geometry = metadata['identity']['geometry']
        if previous_policy == 'reference_support':
            assert geometry.pop(previous_policy)
        else:
            assert geometry[previous_policy] == 'sampled and later unresolved green drift retry RGB luminance'
            geometry[previous_policy] = 'unresolved green drift retries RGB luminance'
        arrays['metadata'] = json.dumps(metadata)
        np.savez(state, **arrays)
        with SERSource(path) as src:
            with pytest.raises(ValueError, match='identity/configuration mismatch'):
                stack_source(src, cfg, resume_from=state)
