import numpy as np
import pytest
from scipy.signal import convolve2d
from planetrecon.operators import (SceneDetectorOperator, translate_cells,
                                   padded_detector_crop, circular_psf_convolve)


def spatial_shift(x, shift):
    out = np.zeros_like(x)
    for y in range(x.shape[0]):
        for col in range(x.shape[1]):
            tx, ty = col+shift[0], y+shift[1]
            for oy in (int(np.floor(ty)), int(np.floor(ty))+1):
                for ox in (int(np.floor(tx)), int(np.floor(tx))+1):
                    if 0 <= oy < x.shape[0] and 0 <= ox < x.shape[1]:
                        out[oy, ox] += x[y, col]*max(0., 1-abs(tx-ox))*max(0., 1-abs(ty-oy))
    return out


def spatial_reference(op, scene):
    channels = [scene] if scene.ndim == 2 else [scene[..., c] for c in range(3)]
    result = []
    b = op.bin_factor
    ox, oy = op.origin_xy
    for channel in channels:
        image = sum(w*convolve2d(spatial_shift(channel, s), p, mode='same')
                    for p, s, w in zip(op.psfs, op.shifts_xy, op.exposure_weights))
        det = np.array([[image[b*y:b*(y+1), b*x:b*(x+1)].sum()
                         for x in range(ox, ox+op.detector_shape[1])]
                        for y in range(oy, oy+op.detector_shape[0])])
        result.append(op.flux*det*op.valid_mask)
    if scene.ndim == 2:
        return result[0]
    if op.cfa_pattern is None:
        return np.stack(result, axis=-1)
    raw = np.zeros(op.detector_shape)
    for y, x in np.ndindex(raw.shape):
        i = 2*((y+oy+op.cfa_offset_xy[1]) % 2)+(x+ox+op.cfa_offset_xy[0]) % 2
        raw[y, x] = result['RGB'.index(op.cfa_pattern[i])][y, x]
    return raw


@pytest.mark.parametrize('pattern', [None, 'RGGB', 'BGGR', 'GRBG', 'GBRG'])
@pytest.mark.parametrize('offset', [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize('kernel_shape', [(3, 5), (4, 2)])
def test_rgb_exposure_cfa_spatial_oracle_and_adjoint(pattern, offset, kernel_shape):
    rng = np.random.default_rng(31)
    psfs = tuple(rng.uniform(size=kernel_shape) for _ in range(2))
    psfs = tuple(p/p.sum() for p in psfs)
    mask = rng.uniform(size=(3, 4)) > .2
    op = SceneDetectorOperator((12, 14, 3), psfs, 2, (2, 1), (3, 4), flux=2.3,
                               shifts_xy=((.375, -.6), (-1.1, .2)), exposure_weights=(.25, .75),
                               valid_mask=mask, cfa_pattern=pattern, cfa_offset_xy=offset)
    scene = rng.normal(size=op.scene_shape)
    residual = rng.normal(size=op.output_shape)
    np.testing.assert_allclose(op.forward(scene), spatial_reference(op, scene), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.vdot(residual, op.forward(scene)), np.vdot(op.adjoint(residual), scene), rtol=1e-12, atol=1e-12)
    direction = rng.normal(size=scene.shape)
    eps = 1e-5
    objective = lambda x: .5*np.sum((op.forward(x)-residual)**2)
    derivative = (objective(scene+eps*direction)-objective(scene-eps*direction))/(2*eps)
    np.testing.assert_allclose(derivative, np.vdot(op.adjoint(op.forward(scene)-residual), direction), rtol=1e-8)


@pytest.mark.parametrize('kernel_shape', [(3, 3), (4, 4)])
def test_units_impulse_constant_limb_and_simulator_compatibility(kernel_shape):
    psf = np.ones(kernel_shape)/np.prod(kernel_shape)
    op = SceneDetectorOperator((40, 48), (psf,), 2, (3, 4), (12, 14), flux=7.)
    constant = np.ones(op.scene_shape)
    np.testing.assert_allclose(op.forward(constant), 28., atol=1e-12)
    impulse = np.zeros(op.scene_shape)
    impulse[20, 22] = 1.
    assert abs(op.forward(impulse).sum()-7.) < 1e-12
    y, x = np.indices(op.scene_shape)
    limb = ((x-12)**2+(y-22)**2 < 100)*(1+.1*np.sin(x))
    np.testing.assert_allclose(op.forward(limb), spatial_reference(op, limb), atol=1e-12)
    square_op = SceneDetectorOperator(op.scene_shape, (psf,), 2, (3, 4), (12, 12), flux=7.)
    np.testing.assert_allclose(square_op.forward(limb), padded_detector_crop(limb, psf, (3, 4), 12, 2, 7.), atol=1e-12)


def test_even_kernel_origin_and_fractional_nyquist_are_explicit():
    psf = np.zeros((4, 4)); psf[2, 2] = 1
    scene = np.zeros((8, 8)); scene[3, 3] = 1
    op = SceneDetectorOperator(scene.shape, (psf,), 1, (0, 0), scene.shape)
    assert np.unravel_index(op.forward(scene).argmax(), scene.shape) == (4, 4)
    nyquist = (-1.)**np.indices(scene.shape).sum(axis=0)
    shifted = translate_cells(nyquist, (.5, 0))
    np.testing.assert_array_equal(shifted[:, 1:], 0.)
    np.testing.assert_array_equal(shifted[:, 0], .5*nyquist[:, 0])


def test_exterior_light_cannot_be_recovered_from_crop_periodic_truth():
    scene = np.zeros((20, 20)); scene[10, 4] = 1
    psf = np.ones((5, 5))/25
    op = SceneDetectorOperator(scene.shape, (psf,), 1, (5, 5), (10, 10))
    expected = op.forward(scene)
    assert expected.sum() > .1
    legacy = circular_psf_convolve(scene[5:15, 5:15], np.pad(psf, ((2, 3), (2, 3))))
    np.testing.assert_array_equal(legacy, 0.)
    assert np.linalg.norm(expected-legacy) > .05


@pytest.mark.parametrize('kwargs', [dict(bin_factor=0), dict(origin_xy=(-1, 0)), dict(flux=np.nan),
    dict(psfs=(np.ones((2, 2)),)), dict(exposure_weights=(.5,)), dict(cfa_pattern='RGGB'),
    dict(valid_mask=np.ones((4, 4))), dict(shifts_xy=((np.inf, 0),))])
def test_invalid_contract_rejected(kwargs):
    args = dict(scene_shape=(8, 8), psfs=(np.ones((1, 1)),), bin_factor=1,
                origin_xy=(0, 0), detector_shape=(4, 4))
    with pytest.raises(ValueError):
        SceneDetectorOperator(**{**args, **kwargs})
