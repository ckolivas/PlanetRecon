"""Known-coordinate controls; these do not assert real-capture improvement."""
import numpy as np
import pytest

from planetrecon.detector import cfa_labels
from planetrecon.pipeline.preprocess import FrameSelection, best_frame_mask
from tools.cfa_local_transport_probe import LocalQuadraticAccumulator, inverse_pull_map


def scene(x, y, kind):
    # Analytic averages over unit detector squares, evaluated before CFA sampling.
    if kind == 'constant':
        return np.broadcast_to([20., 30., 40.], (*x.shape, 3)).copy()
    affine = np.stack([20+.7*x+.3*y, 30-.2*x+.5*y, 40+.3*x-.4*y], axis=-1)
    if kind == 'affine':
        return affine
    if kind == 'quadratic':
        return affine + np.stack([.03*(x*x+1/12)+.02*x*y,
                                   -.02*(y*y+1/12)+.01*x*y,
                                   .01*(x*x+y*y+1/6)-.03*x*y], axis=-1)
    def wave(kx, ky, phase):
        return np.sin(kx*x+ky*y+phase)*np.sinc(kx/(2*np.pi))*np.sinc(ky/(2*np.pi))
    return np.stack([30+4*wave(1/3,0,0)+3*wave(0,1/4,np.pi/2),
                     40+5*wave(1/5,1/5,np.pi/2)+3*wave(1/4,0,0),
                     35+3*wave(1/4,-1/5,0)], axis=-1)



PATTERNS = ['RGGB', 'GRBG', 'GBRG', 'BGGR']


def observations(shape, pattern, kind, index):
    """Analytically inverted row shear; independent of the candidate inverse."""
    yy, xx = np.indices(shape, dtype=float)
    row = np.arange(shape[0], dtype=float)
    shear = .65*np.sin(row/4 + index*.6)
    dx, dy = .22*index-.3, .13*index-.25
    shift = (np.broadcast_to(shear[:, None]+dx, shape), np.full(shape, dy))
    ry = yy-dy
    rx = xx-dx-np.interp(ry, row, shear)
    labels = cfa_labels(*shape, pattern)[..., None] == np.array(list('RGB'))
    raw = (scene(rx, ry, kind)*labels).sum(axis=-1)
    return raw, shift, np.stack([rx, ry], axis=-1)


@pytest.mark.parametrize('pattern', PATTERNS)
@pytest.mark.parametrize('kind', ['constant', 'affine', 'quadratic', 'curved'])
def test_known_local_map_accuracy_with_upper_half_quality_selection(pattern, kind):
    shape = (16, 20)
    qualities = np.array([.2, .8, 1., .3, 1.2, .9])
    measurements = np.zeros((6, 4)); measurements[:, 0] = qualities
    selection = FrameSelection(np.ones(6, bool), measurements, {})
    selected = np.flatnonzero(best_frame_mask(selection, 50, 'quality_range'))
    np.testing.assert_array_equal(selected, [1, 2, 4, 5])
    model = LocalQuadraticAccumulator(shape, pattern)
    for i in selected:
        raw, shift, expected = observations(shape, pattern, kind, int(i))
        actual, valid = inverse_pull_map(shift, shape)
        np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-10, rtol=0)
        model.add(raw, shift, qualities[i])
    after, variance, degree, before = model.finish()
    yy, xx = np.indices(shape, dtype=float)
    truth = scene(xx, yy, kind)
    fits = degree >= (2 if kind == 'quadratic' else 1)
    assert fits[4:-4, 4:-4].all()
    if kind != 'curved':
        np.testing.assert_allclose(after[fits], truth[fits], atol=1e-9, rtol=0)
    else:
        interior = np.s_[4:-4, 4:-4]
        assert np.mean((after[interior]-truth[interior])**2) < np.mean((before.image[interior]-truth[interior])**2)
    assert model.n_used == len(selected)
    assert np.isfinite(after).all() and np.isfinite(variance).all()
    np.testing.assert_array_equal(after[degree <= 0], before.image[degree <= 0])
    np.testing.assert_array_equal(degree >= 0, before.validity)


def test_translation_matches_independent_weighted_least_squares():
    shape = (20, 24)
    shifts = np.array([[0., 0.], [.3, -.4], [-.2, .35]])
    qualities = np.array([.8, 1., 1.2])
    local = LocalQuadraticAccumulator(shape, 'RGGB')
    yy, xx = np.indices(shape, dtype=float)
    labels = cfa_labels(*shape, 'RGGB')
    frames = []
    for i, shift in enumerate(shifts):
        raw = (scene(xx-shift[0], yy-shift[1], 'curved')*(labels[..., None] == np.array(list('RGB')))).sum(axis=-1)
        frames.append(raw)
        local.add(raw, shift, qualities[i])
    after, variance, degree, _ = local.finish()
    # Independent rectangular least-squares solve, without normal equations,
    # inverse maps, KD trees or the candidate's accumulated sufficient statistics.
    for y, x in [(8, 8), (8, 9), (9, 8), (9, 9)]:
        for c, color in enumerate('RGB'):
            design, samples, weight = [], [], []
            for raw, shift, q in zip(frames, shifts, qualities):
                u, v = (xx-shift[0]-x)/4, (yy-shift[1]-y)/4
                take = (abs(u) < 1) & (abs(v) < 1) & (labels == color)
                px, py = u[take], v[take]
                design.extend(np.column_stack([np.ones(len(px)), px, py, px*px, px*py, py*py]))
                samples.extend(raw[take]); weight.extend(q*(1-px*px)**3*(1-py*py)**3)
            phi = np.asarray(design); root_weight = np.sqrt(weight)
            coefficients = np.linalg.pinv(phi*root_weight[:, None])[0]*root_weight
            assert degree[y, x, c] == 2
            np.testing.assert_allclose(after[y, x, c], coefficients@np.asarray(samples), atol=1e-10, rtol=0)
            np.testing.assert_allclose(variance[y, x, c], coefficients@coefficients, atol=1e-12, rtol=0)


def test_raw_impulses_verify_variance_and_colour_isolation():
    shape = (8, 10)
    _, shift, _ = observations(shape, 'RGGB', 'constant', 2)
    outputs = []
    for k in range(np.prod(shape)):
        raw = np.zeros(shape); raw.flat[k] = 1
        model = LocalQuadraticAccumulator(shape, 'RGGB')
        model.add(raw, shift, 1.)
        after, variance, degree, before = model.finish()
        outputs.append(after.ravel())
    operator = np.array(outputs).T
    np.testing.assert_allclose((operator**2).sum(axis=1).reshape((*shape, 3)), variance, atol=1e-12)
    labels = cfa_labels(*shape, 'RGGB').ravel()
    for c, name in enumerate('RGB'):
        np.testing.assert_array_equal(operator[c::3, labels != name], 0)
    np.testing.assert_allclose(operator.sum(axis=1)[before.validity.ravel()], 1., atol=1e-11)


def test_cancellation_and_invalid_map_leave_prior_frame_unchanged():
    shape = (12, 16)
    raw, shift, _ = observations(shape, 'RGGB', 'curved', 1)
    model = LocalQuadraticAccumulator(shape, 'RGGB', chunk_rows=7)
    model.add(raw, shift, .8)
    saved = [v.copy() for v in (model.gram, model.noise, model.rhs, model.sums, model.weights, model.variance_sum)]
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls == 5
    model.should_cancel = cancel
    with pytest.raises(InterruptedError):
        model.add(raw, shift, .9)
    model.should_cancel = None
    yy, xx = np.indices(shape, dtype=float)
    with pytest.raises(ValueError, match='unique contractive inverse'):
        model.add(raw, (-2*xx, yy*0), .9)
    for before, after in zip(saved, (model.gram, model.noise, model.rhs, model.sums, model.weights, model.variance_sum)):
        np.testing.assert_array_equal(before, after)
    assert model.n_used == 1
    model.add(raw, shift, .9)
    assert model.n_used == 2


def test_chunk_size_changes_no_output():
    outputs = []
    for chunk in [1, 19, 256]:
        model = LocalQuadraticAccumulator((8, 10), 'GBRG', chunk_rows=chunk)
        for i in range(2):
            raw, shift, _ = observations(model.shape, model.color, 'curved', i)
            model.add(raw, shift, 1.+i)
        outputs.append(model.finish()[:3])
    for candidate in outputs[1:]:
        for expected, actual in zip(outputs[0], candidate):
            np.testing.assert_array_equal(expected, actual)


def test_output_region_preserves_full_detector_geometry_and_completion():
    shape = (10, 12)
    region = (slice(2, 8), slice(3, 9))
    full = LocalQuadraticAccumulator(shape, 'BGGR')
    crop = LocalQuadraticAccumulator(shape, 'BGGR', region=region)
    for i in range(3):
        raw, shift, _ = observations(shape, 'BGGR', 'curved', i)
        full.add(raw, shift, .8+i/5)
        crop.add(raw, shift, .8+i/5)
    expected, actual = full.finish(), crop.finish()
    for a, b in zip(expected[:3], actual[:3]):
        np.testing.assert_array_equal(a[region], b)
    for key in expected[3].layer_coverage:
        np.testing.assert_array_equal(expected[3].layer_coverage[key][region], actual[3].layer_coverage[key])
