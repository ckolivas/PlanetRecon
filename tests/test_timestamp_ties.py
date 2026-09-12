import numpy as np
import pytest

from planetrecon.detector import cfa_labels
from planetrecon.geometry.pose import capture_timing, source_times_s
from planetrecon.io.ser import SERSource, write_ser
from planetrecon.io.source import ArraySource
from planetrecon.pipeline.baseline import stack_source
from planetrecon.pipeline.geometry_stack import prepare_geometry
from planetrecon.reconstruction import ReconstructionConfig


@pytest.mark.parametrize('dtype', ['i1', 'u1', '<i2', '>u2', '<i4', '>u4', '<i8', '>i8', '<u8', '>u8'])
def test_integer_epochs_match_unbounded_integer_subtraction(dtype):
    dtype = np.dtype(dtype)
    limits = np.iinfo(dtype)
    lo, hi = int(limits.min), int(limits.max)
    # Cover ties, sub-tick differences near large epochs and signed overflow.
    examples = [[lo, lo, lo+1, hi-1, hi], [hi-4, hi-4, hi-3, hi-1, hi]]
    if lo < 0:
        examples.append([lo, -1, 0, 1, hi])
    for values in examples:
        src = ArraySource(np.zeros((len(values), 1, 1)))
        src.timestamps = lambda: np.array(values, dtype=dtype)
        src.timestamp_scale_s = lambda: 1e-7
        with np.errstate(all='raise'):
            actual, origin = source_times_s(src)
        expected = np.array([v-values[0] for v in values], dtype=np.float64)*1e-7
        np.testing.assert_array_equal(actual, expected)
        assert origin == 'measured'


def source(colour='mono', times=(0., 0., .8)):
    yy, xx = np.indices((64, 64))
    radius = np.hypot(xx-32, yy-32)
    disc = radius < 12
    ring = (((xx-32)/25)**2+((yy-32)/10)**2 < 1) & (radius > 14)
    image = 100*disc + 60*ring + disc*(12*np.cos(xx/2)+10*np.sin(yy/3))
    frames = np.stack([image*(1+i*.02) for i in range(3)])
    if colour != 'mono':
        labels = cfa_labels(64, 64, colour)
        frames *= sum(value*(labels == name) for name, value in zip('RGB', (.8, 1., .6)))
    return ArraySource(frames, color_mode=colour, bit_depth=32, timestamps=np.array(times))


def config(**kwargs):
    return ReconstructionConfig(device='cpu', threads=2, frame_preselection=False,
        geometry_mode='saturn', field_center_x=32., field_center_y=32.,
        equatorial_radius_px=12., sub_obs_lat_rad=.4,
        ring_inner_radius_px=16., ring_outer_radius_px=26.,
        surface_rate_rad_s=0., **kwargs)


def test_integer_ser_ties_preserve_recorded_times_and_duration(tmp_path):
    base = 638_900_000_000_000_000
    ticks = np.array([base, base, base+1, base+1, base+100], dtype=np.uint64)
    path = write_ser(tmp_path/'ties.ser', np.ones((5, 8, 8), dtype='u1'), timestamps=ticks)
    with SERSource(path) as src:
        times, origin = source_times_s(src, cadence_s=99.)
        np.testing.assert_allclose(times, [0, 0, 1e-7, 1e-7, 1e-5], rtol=1e-14)
        report = capture_timing(src)
        assert origin == report['origin'] == 'measured'
        assert report['duplicate_intervals'] == 2
        assert report['duration_s'] == pytest.approx(1e-5)
        assert report['minimum_interval_s'] == 0
        assert src.metadata().extras['capture_timing'].value == report


@pytest.mark.parametrize('colour', ['mono', 'RGGB'])
def test_saturn_keeps_distinct_images_with_equal_timestamps(colour):
    cfg = config(field_rate_rad_s=0.)
    tied = stack_source(source(colour), cfg)
    unique = stack_source(source(colour, (0., .4, .8)), cfg)
    assert tied.n_used == unique.n_used == 3 and tied.n_rejected == 0
    assert not tied.incomplete
    np.testing.assert_array_equal(tied.image, unique.image)
    np.testing.assert_array_equal(tied.coverage, unique.coverage)
    assert tied.provenance['geometry']['timestamp_duplicate_intervals'] == 1
    assert any('timestamp_ties:' in warning for warning in tied.warnings)
    assert np.isfinite(tied.image).all()
    if colour != 'mono':
        assert tied.channel_order == 'RGB' and tied.validity.any(axis=(0, 1)).all()


def test_field_rate_fit_excludes_zero_time_intervals(monkeypatch):
    import planetrecon.pipeline.geometry_stack as module
    angles = iter([0., .1, .2])
    monkeypatch.setattr(module, 'estimate_field_angle',
                        lambda *args: {'angle_rad': next(angles), 'degeneracy': []})
    src = source()
    poses, _, diagnostics, _ = prepare_geometry(src, config(),
        [src.read_raw(i) for i in range(3)], sample_indices=[0, 1, 2])
    assert diagnostics['field_rate_rad_s'] == pytest.approx(.125)
    assert poses[0].t_s == poses[1].t_s == 0
    assert poses[0].field_angle_rad == poses[1].field_angle_rad


def test_tied_fit_samples_do_not_invent_a_rate(monkeypatch):
    import planetrecon.pipeline.geometry_stack as module
    monkeypatch.setattr(module, 'estimate_field_angle',
                        lambda *args: {'angle_rad': .1, 'degeneracy': []})
    src = source()
    _, _, diagnostics, _ = prepare_geometry(src, config(),
        [src.read_raw(i) for i in (0, 1)], sample_indices=[0, 1])
    assert diagnostics['field_rate_rad_s'] == 0
    assert 'roll_unconstrained' in diagnostics['degeneracy']
