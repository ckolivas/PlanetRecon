import numpy as np
import pytest
from planetrecon.operators import SceneDetectorOperator
from tools.scene_quadratic import SceneQuadratic
from tools.scene_study import CellBasisOperator
from tools.scene_hessian_diagonal import frame_diagonal, hessian_diagonal


@pytest.mark.parametrize('kernel', [(2, 4), (3, 3)])
@pytest.mark.parametrize('pattern', ['mono', None, 'RGGB', 'BGGR', 'GRBG', 'GBRG'])
def test_integrated_squared_psf_matches_dense_hessian_diagonal(kernel, pattern):
    rng = np.random.default_rng(314)
    shape = (8, 8) if pattern == 'mono' else (8, 8, 3)
    psfs = [rng.uniform(size=kernel) for _ in range(2)]
    psfs = [p/p.sum() for p in psfs]
    op = SceneDetectorOperator(shape, tuple(psfs), 2, (1, 1), (2, 2), exposure_weights=(.2, .8),
                               cfa_pattern=None if pattern == 'mono' else pattern,
                               cfa_offset_xy=(1, 1), flux=2.3, valid_mask=np.array([[True, False], [True, True]]))
    weight = rng.uniform(size=op.output_shape)
    actual = frame_diagonal(op, weight)
    expected = np.zeros(shape)
    for index in np.ndindex(shape):
        x = np.zeros(shape); x[index] = 1.
        expected[index] = np.sum(weight*op.forward(x)**2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_regularization_diagonal_is_not_the_step_majorizer():
    op = SceneDetectorOperator((6, 6), (np.ones((3, 3))/9,), 1, (0, 0), (6, 6))
    p = SceneQuadratic([CellBasisOperator(op, 1)], [np.ones((6, 6))], [1.], ridge=.01, smoothness=.02)
    basis = np.eye(36).reshape((36, 6, 6))
    h = np.stack([p.normal(x).ravel() for x in basis], axis=1)
    actual, info = hessian_diagonal(p, workers=2)
    np.testing.assert_allclose(actual.ravel(), np.diag(h), atol=1e-14)
    assert info['peak_pending_frames'] <= 2
    assert np.linalg.eigvalsh(np.diag(actual.ravel())-h).min() < -.001


def test_unsupported_sampling_and_translation_rejected():
    op = SceneDetectorOperator((8, 8), (np.ones((1, 1)),), 1, (0, 0), (8, 8))
    with pytest.raises(ValueError, match='native'):
        frame_diagonal(CellBasisOperator(op, 2), np.ones((8, 8)))
    shifted = SceneDetectorOperator((8, 8), (np.ones((1, 1)),), 1, (0, 0), (8, 8), shifts_xy=((.2, 0.),))
    with pytest.raises(ValueError, match='translations'):
        frame_diagonal(shifted, np.ones((8, 8)))
