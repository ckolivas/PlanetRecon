"""Reused CUDA pull coordinates retain zero exterior support for every plane."""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates


@pytest.mark.hardware
@pytest.mark.parametrize('shape', [(17, 23), (1, 19), (13, 1)])
@pytest.mark.parametrize('displacement', ['dense', 'zero', 'fractional', 'outside'])
def test_signal_and_support_share_exact_detector_coordinates(shape, displacement):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.backends.torch_accel import TorchBackend
    y, x = np.indices(shape, dtype=float)
    shifts = {'dense': (.7*np.sin(y/3)+.3, .9*np.cos(x/4)-.2),
              'zero': (0., 0.), 'fractional': (-.625, 1.375),
              'outside': (shape[1]+.5, -shape[0]-.5)}
    shift = shifts[displacement]
    rng = np.random.default_rng(82)
    values = [rng.normal(size=shape+(3,)), rng.uniform(size=shape+(3,)),
              rng.normal(size=shape+(3,)), np.ones(shape)]
    tensors = [torch.as_tensor(value, device='cuda', dtype=torch.float64) for value in values]
    actual = TorchBackend._pull_many(tensors, shift)
    for value, result in zip(values, actual):
        def sample(plane):
            return map_coordinates(plane, [y+shift[1], x+shift[0]], order=1,
                                   mode='grid-constant', prefilter=False)
        expected = (sample(value) if value.ndim == 2 else
                    np.stack([sample(value[..., c]) for c in range(value.shape[2])], axis=-1))
        np.testing.assert_allclose(result.cpu().numpy(), expected, atol=2e-14, rtol=2e-14)


@pytest.mark.hardware
@pytest.mark.parametrize('fault', ['shape', 'nan', 'inf'])
def test_shared_pull_rejects_unusable_dense_displacements(fault):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from planetrecon.backends.torch_accel import TorchBackend
    image = torch.ones((13, 17), device='cuda', dtype=torch.float64)
    sx, sy = np.zeros((13, 17)), np.zeros((13, 17))
    if fault == 'shape':
        sy = sy[:-1]
    else:
        sy[2, 3] = np.nan if fault == 'nan' else np.inf
    with pytest.raises(ValueError, match='dense displacement'):
        TorchBackend._pull_many((image, image[..., None].expand(-1, -1, 3)), (sx, sy))
